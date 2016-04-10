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
        if (key == "query_order") {
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

        // does the phrase start with <S>?
        size_t word_idx = 0;
        auto cur_state = cstlm::LMQueryMKN<Model>(&m_cstlm_model, m_ngram_order);
        if (Translate_Moses_2_CSTLMID(phrase.GetWord(0).GetFactor(m_factorType)) == cstlm::PAT_START_SYM) {
            word_idx = 1;
        }
        else {
            // TODO: Is this possible for us? I thought we are always working under the assumption that
            // when we score something it has to start with <S>
        }

        // (1) score the first ngram only
        size_t end_loop = std::min(m_ngram_order, phrase.GetSize());
        for (; word_idx < end_loop; word_idx++) {
            const auto& word = phrase.GetWord(word_idx);
            if (word.IsNonTerminal()) {
                // TODO: what is this? KenLM performs some reset here
            }
            else {
                auto cstlm_tok = Translate_Moses_2_CSTLMID(word);
                fullScore += cur_state.append_symbol(cstlm_tok);
            }
        }
        ngramScore = TransformLMScore(fullScore);
        // (2) score the whole phrase
        for (; word_idx < phrase.GetSize(); word_idx++) {
            const auto& word = phrase.GetWord(word_idx);
            if (word.IsNonTerminal()) {
                // TODO: what is this? KenLM performs some reset here
            }
            else {
                auto cstlm_tok = Translate_Moses_2_CSTLMID(word);
                if (cstlm_tok == cstlm::UNKNOWN_SYM) {
                    oovCount++;
                }
                // TODO: do we still append UNK?
                fullScore += cur_state.append_symbol(cstlm_tok);
            }
        }

        fullScore = TransformLMScore(fullScore);
    }

    virtual FFState* EvaluateWhenApplied(const Hypothesis& hypo, const FFState* previous_state, ScoreComponentCollection* out) const
    {
        const cstlm::LMQueryMKN<Model>& in_state = static_cast<const CstLMState<Model>&>(*previous_state).state;

        std::unique_ptr<CstLMState<Model> > ret(new CstLMState<Model>());
        ret->state = in_state;

        if (!hypo.GetCurrTargetLength()) {
            return ret.release();
        }

        const std::size_t begin = hypo.GetCurrTargetWordsRange().GetStartPos();
        //[begin, end) in STL-like fashion.
        const std::size_t end = hypo.GetCurrTargetWordsRange().GetEndPos() + 1;
        const std::size_t adjust_end = std::min(end, begin + m_ngram_order - 1);

        std::size_t position = begin;

        float score = 0.0f;
        for (; position < adjust_end; ++position) {
            auto cstlm_tok = Translate_Moses_2_CSTLMID(hypo.GetWord(position));
            score += ret->state.append_symbol(cstlm_tok);
        }

        if (hypo.IsSourceCompleted()) {
            // Score end of sentence.
            auto cstlm_tok = cstlm::PAT_END_SYM;
            score += ret->state.append_symbol(cstlm_tok);
        }
        else if (adjust_end < end) {
            // TODO WHAT DO WE NEED THIS FOR?
            // Get state after adding a long phrase.
            // std::vector<lm::WordIndex> indices(m_ngram->Order() - 1);
            // const lm::WordIndex* last = LastIDs(hypo, &indices.front());
            // m_ngram->GetState(&indices.front(), last, ret->state);
        }
        else { // TODO ??? if (state0 != &ret->state) {
            // Short enough phrase that we can just reuse the state.
            // ret->state = *state0;
        }

        score = TransformLMScore(score);
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
        auto itr = m_moses_2_cstlm_id.find(f);
        if (itr != m_moses_2_cstlm_id.end()) {
            return itr->second;
        }
        else {
            return cstlm::UNKNOWN_SYM;
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
