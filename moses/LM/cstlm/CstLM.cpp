#include "CstLM.h"

#include "cstlm/index_types.hpp"

#include "moses/FactorCollection.h"
#include "moses/InputType.h"
#include "moses/TranslationTask.h"

namespace Moses {

template <class t_model>
struct CstLMState : public FFState {
    cstlm::LMQueryMKN<t_model> state;
    virtual size_t hash() const
    {
        return hash_value(state);
    }
    virtual bool operator==(const FFState& o) const
    {
        const CstLMState& other = static_cast<const CstLMState&>(o);
        return state == other.state;
    }
};

template <class Model>
CstLM<Model>::CstLM(const string& line)
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
    auto index_file = m_path + "/index/index-" + sdsl::util::class_to_hash(m_cstlm_model) + ".sdsl";
    cerr << "Loading cstlm index from file " << index_file << endl;
    m_cstlm_model.load(index_file);

    // (3) create alphabet mapping between moses and cstlm
    create_alphabet_mapping();
}

/*
    this method will be called by ReadParameters with the
    extracted key-value pairs.
*/
template <class Model>
void CstLM<Model>::SetParameter(const string& key, const string& value)
{
    if (key == "query_order") {
        m_ngram_order = Scan<uint64_t>(value);
    }
    else if (key == "factor") { // TODO: is this needed?
        m_factorType = Scan<FactorType>(value);
    }
    else {
        LanguageModel::SetParameter(key, value);
    }
}

/*
    map from moses to cstlm alphabets and back
*/
template <class Model>
void CstLM<Model>::create_alphabet_mapping()
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
        const Moses::Factor* factor = collection.AddFactor(str, false);

        // map string
        m_moses_2_cstlm_str[factor] = cstlm_tok_str;
        m_cstlm_str_2_moses[cstlm_tok_str] = factor;
        // map id
        m_moses_2_cstlm_id[factor] = cstlm_tok_id;
        m_cstlm_id_2_moses[cstlm_tok_id] = factor;
    }
}

/*  
    TODO: the main constructor always constructs the <S> state I think?
*/
template <class Model>
const FFState* CstLM<Model>::EmptyHypothesisState(const InputType& /*input*/) const
{
    CstLMState<Model>* ret = new CstLMState<Model>();
    ret->state = cstlm::LMQueryMKN<Model>(m_cstlm_model, m_ngram_order);
    return ret;
}

template <class Model>
void CstLM<Model>::CalcScore(const Phrase& phrase, float& fullScore, float& ngramScore, size_t& oovCount) const
{
    fullScore = 0;
    ngramScore = 0;
    oovCount = 0;

    if (!phrase.GetSize())
        return;

    // size_t position;
    // if (m_beginSentenceFactor == phrase.GetWord(0).GetFactor(m_factorType)) {
    //     scorer.BeginSentence();
    //     position = 1;
    // }
    // else {
    //     position = 0;
    // }

    // size_t ngramBoundary = m_ngram->Order() - 1;

    // size_t end_loop = std::min(ngramBoundary, phrase.GetSize());
    // for (; position < end_loop; ++position) {
    //     const Word& word = phrase.GetWord(position);
    //     if (word.IsNonTerminal()) {
    //         fullScore += scorer.Finish();
    //         scorer.Reset();
    //     }
    //     else {
    //         lm::WordIndex index = TranslateID(word);
    //         scorer.Terminal(index);
    //         if (!index)
    //             ++oovCount;
    //     }
    // }
    // float before_boundary = fullScore + scorer.Finish();
    // for (; position < phrase.GetSize(); ++position) {
    //     const Word& word = phrase.GetWord(position);
    //     if (word.IsNonTerminal()) {
    //         fullScore += scorer.Finish();
    //         scorer.Reset();
    //     }
    //     else {
    //         lm::WordIndex index = TranslateID(word);
    //         scorer.Terminal(index);
    //         if (!index)
    //             ++oovCount;
    //     }
    // }
    // fullScore += scorer.Finish();

    // ngramScore = TransformLMScore(fullScore - before_boundary);
    // fullScore = TransformLMScore(fullScore);
}

template <class Model>
FFState* CstLM<Model>::EvaluateWhenApplied(const Hypothesis& hypo, const FFState* previous_state, ScoreComponentCollection* out) const
{
    const cstlm::LMQueryMKN<Model>& in_state = static_cast<const cstlm::LMQueryMKN<Model>&>(*ps).state;

    std::unique_ptr<CstLMState> ret(new CstLMState());

    if (!hypo.GetCurrTargetLength()) {
        ret->state = in_state;
        return ret.release();
    }

    const std::size_t begin = hypo.GetCurrTargetWordsRange().GetStartPos();
    //[begin, end) in STL-like fashion.
    const std::size_t end = hypo.GetCurrTargetWordsRange().GetEndPos() + 1;
    const std::size_t adjust_end = std::min(end, begin + m_ngram->Order() - 1);

    std::size_t position = begin;

    typename Model::State aux_state;
    typename Model::State *state0 = &ret->state, *state1 = &aux_state;

    float score = m_ngram->Score(in_state, TranslateID(hypo.GetWord(position)), *state0);
    float score++ position;
    for (; position < adjust_end; ++position) {
        score += m_ngram->Score(*state0, TranslateID(hypo.GetWord(position)), *state1);
        std::swap(state0, state1);
    }

    if (hypo.IsSourceCompleted()) {
        // Score end of sentence.
        std::vector<lm::WordIndex> indices(m_ngram->Order() - 1);
        const lm::WordIndex* last = LastIDs(hypo, &indices.front());
        score += m_ngram->FullScoreForgotState(&indices.front(), last, m_ngram->GetVocabulary().EndSentence(), ret->state).prob;
    }
    else if (adjust_end < end) {
        // Get state after adding a long phrase.
        std::vector<lm::WordIndex> indices(m_ngram->Order() - 1);
        const lm::WordIndex* last = LastIDs(hypo, &indices.front());
        m_ngram->GetState(&indices.front(), last, ret->state);
    }
    else if (state0 != &ret->state) {
        // Short enough phrase that we can just reuse the state.
        ret->state = *state0;
    }

    score = TransformLMScore(score);

    // if (OOVFeatureEnabled()) {
    //     std::vector<float> scores(2);
    //     scores[0] = score;
    //     scores[1] = 0.0;
    //     out->PlusEquals(this, scores);
    // }
    // else {
    //     out->PlusEquals(this, score);
    // }

    return ret.release();
}

/* define the different model types */
namespace cstlm {
    using charlm = index_succinct<default_cst_byte_type>;
    using wordlm = index_succinct<default_cst_int_type>;
}

/* Taken from Ken.cpp. Not sure if needed Instantiate CstLM here.  
 * Tells the compiler to generate code
 * for the instantiations' non-inline member functions in this file.
 * Otherwise, depending on the compiler, those functions may not be present
 * at link time.
 */

template class CstLM<cstlm::charlm>;
template class CstLM<cstlm::wordlm>;