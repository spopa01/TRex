#ifndef __PARSER_H__
#define __PARSER_H__

class RulePkt;
class SubPkt;

RulePkt* buildRule( std::string const& rule );
SubPkt* buildPlainSubscription( RulePkt* rule );

#endif//__PARSER_H__
