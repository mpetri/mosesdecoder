// $Id$
#pragma once

#include <unordered_map>

#include "moses/LM/Base.h"
#include "moses/Hypothesis.h"
#include "moses/TypeDef.h"
#include "moses/Word.h"
#include "moses/Factor.h"

namespace Moses {

//class LanguageModel;
class FFState;

template <class Model>
class CstLM : public LanguageModel {
public:
    CstLM(const std::string& line);
    ~CstLM();

    /* called by baseclass so we can read our parameters */
    void SetParameter(const std::string& key, const std::string& value);

    virtual const FFState* EmptyHypothesisState(const InputType& input);

    /* calc total unweighted LM score of this phrase and return score via arguments.
   * Return scores should always be in natural log, regardless of representation with LM implementation.
   * Uses GetValue() of inherited class.
   * \param fullScore scores of all unigram, bigram... of contiguous n-gram of the phrase
   * \param ngramScore score of only n-gram of order m_nGramOrder
   * \param oovCount number of LM OOVs
   */
    virtual void CalcScore(const Phrase& phrase, float& fullScore, float& ngramScore, std::size_t& oovCount) const = 0;

    virtual void CalcScoreFromCache(const Phrase& phrase, float& fullScore, float& ngramScore, std::size_t& oovCount) const
    {
    }

    virtual void IssueRequestsFor(Hypothesis& hypo,
        const FFState* input_state)
    {
    }
    virtual void sync()
    {
    }
    virtual void SetFFStateIdx(int state_idx)
    {
    }

    lm::WordIndex TranslateID(const Word& word) const
    {
        std::size_t factor = word.GetFactor(m_factorType)->GetId();
        return (factor >= m_lmIdLookup.size() ? 0 : m_lmIdLookup[factor]);
    }

protected:
    FactorType m_factorType;
    Model m_cstlm_model;
    uint64_t m_ngram_order; // query ngram order

    // alphabet mapping between moses and cstlm
    std::unordered_map<const Moses::Factor*, uint64_t> m_moses_2_cstlm_id;
    std::unordered_map<uint64_t, const Moses::Factor*> m_cstlm_id_2_moses;
    std::unordered_map<std::string, const Moses::Factor*> m_cstlm_str_2_moses;
    std::unordered_map<const Moses::Factor*, std::string> m_moses_2_cstlm_str;
};

} // namespace Moses
