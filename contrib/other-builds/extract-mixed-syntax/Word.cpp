/*
 * Word.cpp
 *
 *  Created on: 18 Feb 2014
 *      Author: s0565741
 */
#include <limits>
#include "Word.h"

using namespace std;

Word::Word(const std::string &str)
:m_highestAlignment(-1)
,m_lowestAlignment(numeric_limits<int>::max())
{
	// TODO Auto-generated constructor stub

}

Word::~Word() {
	// TODO Auto-generated destructor stub
}

void Word::AddAlignment(int align)
{
	m_alignment.insert(align);
	if (align > m_highestAlignment) {
		m_highestAlignment = align;
	}
	if (align < m_lowestAlignment) {
		m_lowestAlignment = align;
	}
}
