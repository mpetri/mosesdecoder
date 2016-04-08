// $Id$
#pragma once

#include "moses/LM/Base.h"
#include "moses/Hypothesis.h"
#include "moses/TypeDef.h"
#include "moses/Word.h"

namespace Moses
{

//class LanguageModel;
class FFState;

template <class Model> 
class CstLM : public LanguageModel
{
public:
  CstLM(const std::string &line);
  ~CstLM();
  
  /* called by baseclass so we can read our parameters */
  void SetParameter(const std::string& key, const std::string& value);

  virtual const FFState* EmptyHypothesisState(const InputType &input) const = 0;

  /* calc total unweighted LM score of this phrase and return score via arguments.
   * Return scores should always be in natural log, regardless of representation with LM implementation.
   * Uses GetValue() of inherited class.
   * \param fullScore scores of all unigram, bigram... of contiguous n-gram of the phrase
   * \param ngramScore score of only n-gram of order m_nGramOrder
   * \param oovCount number of LM OOVs
   */
  virtual void CalcScore(const Phrase &phrase, float &fullScore, float &ngramScore, std::size_t &oovCount) const = 0;

  virtual void CalcScoreFromCache(const Phrase &phrase, float &fullScore, float &ngramScore, std::size_t &oovCount) const {
  }

  virtual void IssueRequestsFor(Hypothesis& hypo,
                                const FFState* input_state) {
  }
  virtual void sync() {
  }
  virtual void SetFFStateIdx(int state_idx) {
  }

protected:
  Model m_cstlm_model;
  uint64_t m_ngram_order;
};

} // namespace Moses
