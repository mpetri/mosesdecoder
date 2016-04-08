#include "CstLM.h"

#include "index_types.hpp"

#include "moses/FactorCollection.h"
#include "moses/InputType.h"
#include "moses/TranslationTask.h"


namespace Moses
{

template<class Model>
CstLM<Model>::CstLM(const string &line) : LanguageModel(line),
{
  // (1) call baseclass parameter reader which will call
  // our SetParameter function
  ReadParameters();

  // (2) load the index 
  auto index_file = m_path + "/index/index-" + sdsl::util::class_to_hash(m_cstlm_model) + ".sdsl";
  cerr << "Loading cstlm index from file " << index_file << endl;
  m_cstlm_model.load(index_file);
}

/*
    this class will be called by ReadParameters with the
    extracted key-value pairs.
*/
template<class Model>
void CstLM<Model>::SetParameter(const string& key, const string& value)
{
  if (key == "query_order") {
    m_ngram_order = Scan<uint64_t>(value);
  } else {
    LanguageModel::SetParameter(key, value);
  }
}


/* define the different model types */
namespace cstlm {
    using charlm = index_succinct<default_cst_byte_type>;
    using wordlm = index_succinct<default_cst_int_type>;
}

/* Taken from Ken.cpp Instantiate CstLM here.  Tells the compiler to generate code
 * for the instantiations' non-inline member functions in this file.
 * Otherwise, depending on the compiler, those functions may not be present
 * at link time.
 */
template class CstLM<cstlm::charlm>;
template class CstLM<cstlm::wordlm>;