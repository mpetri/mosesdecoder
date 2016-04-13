// $Id$
#pragma once

#include <unordered_map>

#include "moses/LM/Base.h"
#include "moses/Hypothesis.h"
#include "moses/TypeDef.h"
#include "moses/Word.h"
#include "moses/Factor.h"

#include "cstlm/constants.hpp"
#include "cstlm/index_types.hpp"
#include "cstlm/query.hpp"

namespace Moses {

class FFState;

template <class t_model>
struct CstLMState : public FFState {
    cstlm::LMQueryMKN<t_model> state;
    virtual size_t hash() const
    {
        return state.hash();
    }
    virtual bool operator==(const FFState& o) const
    {
        const CstLMState& other = static_cast<const CstLMState&>(o);
        return state == other.state;
    }
};

template <class Model>
class CstLM : public LanguageModel {
public:
    CstLM(const std::string& line)
        : LanguageModel(line)
    {
        // (1) call baseclass parameter reader which will call
        // our SetParameter function
        ReadParameters();
        // TODO: do we need this??
        if (m_factorType == NOT_FOUND) {
            m_factorType = 0;
        }

        // (2) load the index
        std::string index_file = m_collection_dir + "/index/index-" + sdsl::util::class_to_hash(m_cstlm_model) + ".sdsl";
        std::cerr << "loading cstlm index from file " << index_file << std::endl;
        if (cstlm::utils::file_exists(index_file)) {
            std::cerr << "loading cstlm index from file '" << index_file << "'";
            sdsl::load_from_file(m_cstlm_model, index_file);
        }
        else {
            std::cerr << "cstlm index " << index_file << " does not exist. build it first";
            exit(EXIT_FAILURE);
        }

        // (3) create alphabet mapping between moses and cstlm
        std::cerr << "create alphabet mapping between moses and cstlm " << std::endl;
        create_alphabet_mapping();
    }

    ~CstLM()
    {
    }

    /* called by baseclass so we can read our parameters */
    void SetParameter(const std::string& key, const std::string& value)
    {
        if (key == "order") {
            m_ngram_order = Scan<uint64_t>(value);
        }
        else if (key == "path") { // TODO: is this needed?
            m_collection_dir = value;
        }
        else if (key == "factor") { // TODO: is this needed?
            m_factorType = Scan<FactorType>(value);
        }
        else {
            LanguageModel::SetParameter(key, value);
        }
    }

    virtual const FFState* EmptyHypothesisState(const InputType& input) const
    {
        CstLMState<Model>* ret = new CstLMState<Model>();
        ret->state = cstlm::LMQueryMKN<Model>(&m_cstlm_model, m_ngram_order);
        return ret;
    }

    void create_alphabet_mapping()
    {
        auto& collection = FactorCollection::Instance();
        for (const auto& token : m_cstlm_model.vocab) {
            auto cstlm_tok_str = token.first;
            auto cstlm_tok_id = token.second;

            /*
             * All Factors in moses are accessed and created by a FactorCollection.
             * By enforcing this strict creation processes (ie, forbidding factors
             * from being created on the stack, etc), their memory addresses can
             * be used as keys to uniquely identify them.
             * Only 1 FactorCollection object should be created.
            */
            const Moses::Factor* factor = collection.AddFactor(cstlm_tok_str, false);

            // map string
            m_moses_2_cstlm_str[factor] = cstlm_tok_str;
            m_cstlm_str_2_moses[cstlm_tok_str] = factor;
            // map id
            m_moses_2_cstlm_id[factor] = cstlm_tok_id;
            m_cstlm_id_2_moses[cstlm_tok_id] = factor;
        }
    }

    /* calc total unweighted LM score of this phrase and return score via arguments.
   * Return scores should always be in natural log, regardless of representation with LM implementation.
   * Uses GetValue() of inherited class.
   * \param fullScore scores of all unigram, bigram... of contiguous n-gram of the phrase
   * \param ngramScore score of only n-gram of order m_nGramOrder
   * \param oovCount number of LM OOVs
   */
    virtual void CalcScore(const Phrase& phrase, float& fullScore, float& ngramScore, std::size_t& oovCount) const
    {
        fullScore = 0;
        ngramScore = 0;
        oovCount = 0;

        // empty phrase?
        if (!phrase.GetSize())
            return;

        auto cur_state = cstlm::LMQueryMKN<Model>(&m_cstlm_model, m_ngram_order, false);
        // (1) score the first ngram only
        for (size_t word_idx = 0; word_idx < phrase.GetSize(); word_idx++) {
            const auto& word = phrase.GetWord(word_idx);
            if (word.IsNonTerminal()) {
                std::cerr << "\n CstLM::CalcScore: word.IsNonTerminal == true -> THIS SHOULD NOT HAPPEN!!!" << std::endl;
                // TODO: what is this? KenLM performs some reset here
            }
            else {
                if (!IS_IN_VOCAB(word)) {
                    oovCount++;
                }
                fullScore += Append_Token(cur_state, word);
            }
        }
        fullScore = TransformLMScore(fullScore);
    }

    virtual FFState* EvaluateWhenApplied(const Hypothesis& hypo, const FFState* previous_state, ScoreComponentCollection* out) const
    {
        std::cerr << "CstLM::EvaluateWhenApplied" << std::endl;
        const cstlm::LMQueryMKN<Model>& in_state = static_cast<const CstLMState<Model>&>(*previous_state).state;

        std::unique_ptr<CstLMState<Model> > ret(new CstLMState<Model>());
        ret->state = in_state; // make a copy of the in_state
        if (!hypo.GetCurrTargetLength()) {
            return ret.release();
        }
        const std::size_t begin = hypo.GetCurrTargetWordsRange().GetStartPos();
        //[begin, end) in STL-like fashion.
        const std::size_t end = hypo.GetCurrTargetWordsRange().GetEndPos() + 1;
        float score = 0.0f;
        for (auto position = begin; position < end; ++position) {
            score += Append_Token(ret->state, hypo.GetWord(position));
        }
        if (hypo.IsSourceCompleted()) {
            // Score end of sentence.
            score += ret->state.append_symbol(cstlm::PAT_END_SYM);
        }
        score = TransformLMScore(score);
        std::cerr << "transformed score = " << score << std::endl;
        if (OOVFeatureEnabled()) { // Taken from Ken.cpp
            std::vector<float> scores(2);
            scores[0] = score;
            scores[1] = 0;
            out->PlusEquals(this, scores);
        }
        else {
            out->PlusEquals(this, score);
        }
        return ret.release();
    }
    virtual FFState* EvaluateWhenApplied(const ChartHypothesis& cur_hypo, int featureID, ScoreComponentCollection* accumulator) const
    {
        return nullptr;
    }
    virtual FFState* EvaluateWhenApplied(const Syntax::SHyperedge& hyperedge, int featureID, ScoreComponentCollection* accumulator) const
    {
        return nullptr;
    }
    uint64_t Translate_Moses_2_CSTLMID(const Word& word) const
    {
        const auto& factor = word.GetFactor(m_factorType);
        return Translate_Moses_2_CSTLMID(factor);
    }

    uint64_t Translate_Moses_2_CSTLMID(const Moses::Factor* f) const
    {
        std::cerr << "TranslateID(" << f->GetString() << ")" << std::endl;
        auto itr = m_moses_2_cstlm_id.find(f);
        if (itr != m_moses_2_cstlm_id.end()) {
            return itr->second;
        }
        else {
            return cstlm::UNKNOWN_SYM;
        }
    }

    bool IS_IN_VOCAB(const Word& word) const
    {
        const auto& factor = word.GetFactor(m_factorType);
        return IS_IN_VOCAB(factor);
    }

    bool IS_IN_VOCAB(const Moses::Factor* f) const
    {
        if (Model::byte_alphabet == false) {
            return m_moses_2_cstlm_id.find(f) != m_moses_2_cstlm_id.end();
        }
        else {
            return true;
        }
    }

    template <class t_state>
    float Append_Token(t_state& state, const Word& word) const
    {
        const auto& factor = word.GetFactor(m_factorType);
        return Append_Token(state, factor);
    }

    template <class t_state>
    float Append_Token(t_state& state, const Moses::Factor* f) const
    {
        if (Model::byte_alphabet == false) {
            auto cstlm_tok = Translate_Moses_2_CSTLMID(f);
            return state.append_symbol(cstlm_tok);
        }
        else {
            // (1) append delim
            float score = 0;
            if (state.empty()) {
                auto delim_id = m_cstlm_model.vocab.token2id(" ");
                score += state.append_symbol(delim_id);
            }
            // (2) append bytes in factor string
            const auto& factor_str = f->GetString();
            for (const auto& sym : factor_str) {
                auto delim_id = m_cstlm_model.vocab.token2id(std::string(1, sym), cstlm::UNKNOWN_SYM);
                score += state.append_symbol(cstlm::UNKNOWN_SYM);
            }
            return score;
        }
    }

    virtual bool IsUseable(const FactorMask& mask) const
    {
        return mask[m_factorType];
    }

protected:
    FactorType m_factorType;
    Model m_cstlm_model;
    uint64_t m_ngram_order; // query ngram order
    std::string m_collection_dir;

    // alphabet mapping between moses and cstlm
    std::unordered_map<const Moses::Factor*, typename Model::value_type> m_moses_2_cstlm_id;
    std::unordered_map<typename Model::value_type, const Moses::Factor*> m_cstlm_id_2_moses;
    std::unordered_map<std::string, const Moses::Factor*> m_cstlm_str_2_moses;
    std::unordered_map<const Moses::Factor*, std::string> m_moses_2_cstlm_str;
};

} // namespace Moses
