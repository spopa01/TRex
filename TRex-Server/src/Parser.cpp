#include <boost/spirit/include/qi.hpp>
#include <boost/fusion/adapted.hpp>
#include <boost/optional.hpp>
#include <boost/variant.hpp>

#include <iostream>
#include <string>
#include <vector>

#include <utility>

namespace qi = boost::spirit::qi;
namespace ascii = boost::spirit::ascii;

/*
Just as a note: here (at the moment) I am only trying to match as much as possible the original grammar,
but I believe that the original grammar requires refactoring / improving.
*/

typedef boost::variant<std::string, int, float, bool> static_value;

typedef std::string event_name;
typedef std::string attribute_name;
typedef std::string parameter_name;

// event_declaration = int_ >> "=>" >> event_name;
struct event_declaration{
  int id_;
  event_name event_name_;
};

BOOST_FUSION_ADAPT_STRUCT(
  event_declaration,

  (int, id_)
  (event_name, event_name_)
)

// attribute_type = ( "string" | "int" | "float" | "bool" )
// attribute_declaration = attribute_name >> ':' attribute_type;
struct attribute_declaration{
  enum attribute_type { string_type, int_type, float_type, bool_type };

  attribute_name attribute_name_;
  attribute_type attribute_type_;
};

BOOST_FUSION_ADAPT_STRUCT(
  attribute_declaration,

  (attribute_name, attribute_name_)
  (attribute_declaration::attribute_type, attribute_type_)
)

// event_definition = event_name >> '(' >> -( attribute_declaration % ',' ) >> ')';
struct event_definition{
  event_name event_name_;
  boost::optional<std::vector<attribute_declaration> > attributes_declaration_;
};

BOOST_FUSION_ADAPT_STRUCT(
  event_definition,

  (event_name, event_name_)
  (boost::optional<std::vector<attribute_declaration> >, attributes_declaration_)
)

// attribute_reference = event_name >> '.' >> attribute_name;
struct attribute_reference{
  event_name event_name_;
  attribute_name attribute_name_;
};

BOOST_FUSION_ADAPT_STRUCT(
  attribute_reference,

  (event_name, event_name_)
  (attribute_name, attribute_name_)
)

// parameter_mapping = attribute_name >> "=>" >> parameter_name;
struct parameter_mapping{
  attribute_name attribute_name_;
  parameter_name parameter_name_;
};

BOOST_FUSION_ADAPT_STRUCT(
  parameter_mapping,

  (attribute_name, attribute_name_)
  (parameter_name, parameter_name_)
)

// parameter_atom = ( attribute_reference | parameter_name | static_value )
typedef boost::variant< attribute_reference, parameter_name, static_value  > parameter_atom;

struct expression;
struct term;
struct factor;

struct aggregate_atom;
typedef boost::variant< parameter_atom, boost::recursive_wrapper<aggregate_atom>, 
                        boost::recursive_wrapper<factor>, boost::recursive_wrapper<expression> > atom;

struct factor{
  enum sign { pos_sgn, neg_sgn };

  sign sign_; // +,-
  atom atom_;
};

BOOST_FUSION_ADAPT_STRUCT(
  factor,

  (factor::sign, sign_)
  (atom, atom_)
)

struct term{
  enum op_type { mul_op, div_op, add_op, sub_op };

  op_type op_; // *, /, +, -
  atom atom_;
};

BOOST_FUSION_ADAPT_STRUCT(
  term,

  (term::op_type, op_)
  (atom, atom_)
)

struct expression{
  atom first_;
  std::vector<term> rest_;
};

BOOST_FUSION_ADAPT_STRUCT(
  expression,

  (atom, first_)
  (std::vector<term>, rest_)
)

// within_reference = ( "within" >> uint_ >> delta_type >> "from" >> event_name )
struct within_reference{
  enum delta_type{ millisecs, secs, mins, hours, days };

  unsigned int delta_;
  delta_type delta_type_;
  event_name event_name_;
};

BOOST_FUSION_ADAPT_STRUCT(
  within_reference,

  (unsigned int, delta_)
  (within_reference::delta_type, delta_type_)
  (event_name, event_name_)
)

// between_reference = ( "between" >> event_name >> "and" >> event_name )
struct between_reference{
  event_name fst_event_name_;
  event_name snd_event_name_;
};

BOOST_FUSION_ADAPT_STRUCT(
  between_reference,

  (event_name, fst_event_name_)
  (event_name, snd_event_name_)
)

// predicate_reference = ( within_reference | between_reference )
typedef boost::variant<within_reference, between_reference> predicate_reference;

// TRIGGER PATTERN DEFINITION SECTION

// op_type = ( "==", ">", ">=", "<", "<=", "!=" )
// simple_attribute_constraint = ( atrribute_name >> op_type >> static_value );
struct simple_attribute_constraint{
  enum op_type{ eq_op, gt_op, ge_op, lt_op, le_op, neq_op };

  attribute_name attribute_name_;
  op_type op_;
  static_value static_value_;
};

BOOST_FUSION_ADAPT_STRUCT(
  simple_attribute_constraint,

  (attribute_name, attribute_name_)
  (simple_attribute_constraint::op_type, op_)
  (static_value, static_value_)
)

// complex_attribute_constraint = ( '[' >> attribute_type >> ']' >> attribute_name >> op_type >> expression )
struct complex_attribute_constraint{
  attribute_declaration::attribute_type attribute_type_;
  attribute_name attribute_name_;
  simple_attribute_constraint::op_type op_;
  expression expression_;
};

BOOST_FUSION_ADAPT_STRUCT(
  complex_attribute_constraint,

  (attribute_declaration::attribute_type, attribute_type_)
  (attribute_name, attribute_name_)
  (simple_attribute_constraint::op_type, op_)
  (expression, expression_)
)

// attribute_constraint = simple_attribute_constraint | complex_attribute_constraint;
typedef boost::variant< simple_attribute_constraint, complex_attribute_constraint > attribute_constraint;

// aggregate_atom = aggregation_type >> '(' >> attribute_reference >> '(' >> -(attribute_constraint % ',')  >> ')' >> ')' >> predicate_reference;
struct aggregate_atom{
  enum aggregation_type{ avg_agg, sum_agg, min_agg, max_agg, count_agg };

  aggregation_type aggregation_type_;
  attribute_reference attribute_reference_;
  boost::optional<std::vector<attribute_constraint> > attribute_constraints_;
  predicate_reference reference_;
};

BOOST_FUSION_ADAPT_STRUCT(
  aggregate_atom,

  (aggregate_atom::aggregation_type, aggregation_type_)
  (attribute_reference, attribute_reference_)
  (boost::optional<std::vector<attribute_constraint> >, attribute_constraints_)
  (predicate_reference, reference_)
)

// predicate_parameter = ( parameter_mapping, simple_attribute_constraint | complex_attribute_constraint );
typedef boost::variant<parameter_mapping, simple_attribute_constraint, complex_attribute_constraint> predicate_parameter;

// predicate = ( event_name >> '(' >> -( predicate_parameter % ',' ) >> ')' >> -alias );
struct predicate{
  event_name event_name_;
  boost::optional<std::vector<predicate_parameter> > predicate_parameters_;
  boost::optional<event_name> alias_;
};

BOOST_FUSION_ADAPT_STRUCT(
  predicate,

  (event_name, event_name_)
  (boost::optional<std::vector<predicate_parameter> >, predicate_parameters_)
  (boost::optional<event_name>, alias_)
)

// selection_policy = ( "each" | "first" | "last" );
// positive_predicate = ( "and" >> selection_policy >> predicate >> within_reference );
struct positive_predicate{
  enum selection_policy { each_policy, first_policy, last_policy };
  selection_policy selection_policy_;

  predicate predicate_;
  within_reference reference_;
};

BOOST_FUSION_ADAPT_STRUCT(
  positive_predicate,

  (positive_predicate::selection_policy, selection_policy_)
  (predicate, predicate_)
  (within_reference, reference_)
)

// negatine_predicate = ( "and not" >> predicate >> predicate_reference );
struct negative_predicate{
  predicate predicate_;
  predicate_reference reference_;
};

BOOST_FUSION_ADAPT_STRUCT(
  negative_predicate,

  (predicate, predicate_)
  (predicate_reference, reference_)
)

// predicate_pattern = ( negative_predicate | positive_predicate )
typedef boost::variant<positive_predicate, negative_predicate> pattern_predicate;

// trigger_pattern_definition = ( predicate >> -( pattern_predicate % ',' ) );
struct trigger_pattern_definition{
  predicate trigger_predicate_;
  boost::optional<std::vector<pattern_predicate> > pattern_predicates_;
};

BOOST_FUSION_ADAPT_STRUCT(
  trigger_pattern_definition,

  (predicate, trigger_predicate_)
  (boost::optional<std::vector<pattern_predicate> >, pattern_predicates_)
)

// ATTRIBUTES DEFINITION SECTION

// simple_attribute_definition = ( attribute_name >> "::=" >> static_value )
struct simple_attribute_definition{
  attribute_name attribute_name_;
  static_value static_value_;
};

BOOST_FUSION_ADAPT_STRUCT(
  simple_attribute_definition,

  (attribute_name, attribute_name_)
  (static_value, static_value_)
)

// complex_attribute_definition = ( attribute_name >> ":=" >> expression )
struct complex_attribute_definition{
  attribute_name attribute_name_;
  expression expression_;
};

BOOST_FUSION_ADAPT_STRUCT(
  complex_attribute_definition,

  (attribute_name, attribute_name_)
  (expression, expression_)
)

// attribute_definition = ( simple_attribute_definition | complex_atrribute_definition )
typedef boost::variant<simple_attribute_definition, complex_attribute_definition> attribute_definition;

// RULE DEFINITION SECTION

/*
rule %=   ( "assign" >> (event_declaration % ',') )
     >>   ( "define" >> event_definition )
     >>   ( "from" >> trigger_pattern_definition )
     >> - ( "where" >> ( atrributes_definition % ',' ) )
     >> - ( "consuming" >> ( event_name % ',' ) )
     >> ';';
*/
struct tesla_rule{
  std::vector<event_declaration> events_declaration_;
  event_definition event_definition_; 
  trigger_pattern_definition trigger_pattern_;
  boost::optional<std::vector<attribute_definition> > attributes_definition_;
  boost::optional<std::vector<event_name> > events_to_consume_;
};

BOOST_FUSION_ADAPT_STRUCT(
  tesla_rule,

  (std::vector<event_declaration>, events_declaration_)
  (event_definition, event_definition_)
  (trigger_pattern_definition, trigger_pattern_)
  (boost::optional<std::vector<attribute_definition> >, attributes_definition_)
  (boost::optional<std::vector<event_name> >, events_to_consume_)
)

//--

template<typename It>
struct tesla_grammar : qi::grammar<It, tesla_rule(), ascii::space_type>{
  tesla_grammar() : tesla_grammar::base_type(tesla_rule_){
    using namespace qi;

    //-- 0 & 1

    strlit_ =  ('"' >> *~char_('"') >> '"');

    event_name_ = (char_("A-Z") >> *char_("0-9a-zA-Z"));
    attribute_name_ = (char_("a-z") >> *char_("0-9a-zA-Z"));
    parameter_name_ = (char_('$') >> attribute_name_);

    static_value_ = ( strlit_ | int_ | float_ | bool_ );

    event_declaration_ = int_ >> "=>" >> event_name_;
    events_declaration_ = (event_declaration_ % ',');

    type_token_.add
      ("string", attribute_declaration::string_type)
      ("int", attribute_declaration::int_type)
      ("float", attribute_declaration::float_type)
      ("bool", attribute_declaration::bool_type);
    attribute_type_ = type_token_;
    attribute_declaration_ = attribute_name_ >> ':' >> attribute_type_;

    event_definition_ = event_name_ >> '(' >> -(attribute_declaration_ % ',') >> ')';

    //-- 2

    parameter_mapping_ = (attribute_name_ >> "=>" >> parameter_name_);
    
    attribute_reference_ = (event_name_ >> '.' >> attribute_name_);

    parameter_atom_ = ( attribute_reference_ | parameter_name_ | static_value_ );
    
    factor_sign_token_.add
      ("-", factor::neg_sgn)
      ("+", factor::pos_sgn);
   factor_sign_ = factor_sign_token_;

    term_mul_div_op_type_token_.add
      ("*", term::mul_op)
      ("/", term::div_op);
    term_mul_div_op_type_ = term_mul_div_op_type_token_;
    term_add_sub_op_type_token_.add
      ("+", term::add_op)
      ("-", term::sub_op);
    term_add_sub_op_type_ = term_add_sub_op_type_token_;

    expression_ = term_ >> *( term_add_sub_op_type_ >> term_ );
    term_ = factor_ >> *( term_mul_div_op_type_ >> factor_ );
    factor_ = parameter_atom_ | aggregate_atom_ | '(' >> expression_ >> ')' | (factor_sign_ >> factor_);
 
    delta_type_token_.add
      ("millisecs", within_reference::millisecs)
      ("secs", within_reference::secs)
      ("mins", within_reference::mins)
      ("hours", within_reference::hours)
      ("days", within_reference::days);
    delta_type_ = delta_type_token_;
    within_reference_ = lexeme["within"] >> uint_ >> delta_type_ >> lexeme["from"] >> event_name_;
    between_reference_ = lexeme["between"] >> event_name_ >> lexeme["and"] >> event_name_;

    predicate_reference_ = ( within_reference_ | between_reference_ );

    constr_op_type_token_.add
      ("==", simple_attribute_constraint::eq_op)
      (">", simple_attribute_constraint::gt_op)
      (">=", simple_attribute_constraint::ge_op)
      ("<", simple_attribute_constraint::lt_op)
      ("<=", simple_attribute_constraint::le_op)
      ("!=", simple_attribute_constraint::neq_op);
    constr_op_type_ = constr_op_type_token_;
    simple_attribute_constraint_ = ( attribute_name_ >> constr_op_type_ >> static_value_ );
    complex_attribute_constraint_ = ( '[' >> attribute_type_ >> ']' >> attribute_name_ >> constr_op_type_ >> expression_ );
    attribute_constraint_ = ( simple_attribute_constraint_ | complex_attribute_constraint_ );

    agg_type_token_.add
      ("AVG", aggregate_atom::avg_agg)
      ("SUM", aggregate_atom::sum_agg)
      ("MIN", aggregate_atom::min_agg)
      ("MAX", aggregate_atom::max_agg)
      ("COUNT", aggregate_atom::count_agg);
    agg_type_ = agg_type_token_;
    aggregate_atom_ = (agg_type_ >> '(' >> attribute_reference_ >> '(' >> -( attribute_constraint_ % ',' )  >> ')'  >> ')' >> predicate_reference_ );

    predicate_parameter_ = (parameter_mapping_ | simple_attribute_constraint_ | complex_attribute_constraint_);
    predicate_ = event_name_ >> '(' >> -(predicate_parameter_ % ',') >> ')' >> -(lexeme["as"] >> event_name_);

    selection_policy_token_.add
      ("each", positive_predicate::each_policy)
      ("first", positive_predicate::first_policy)
      ("last", positive_predicate::last_policy);
    selection_policy_ = selection_policy_token_;

    positive_predicate_ = lexeme ["and"] >> selection_policy_ >> predicate_ >> within_reference_;
    negative_predicate_ = lexeme ["and not"] >> predicate_ >> predicate_reference_;

    trigger_pattern_definition_ = predicate_ >> *( positive_predicate_ | negative_predicate_ );

    //-- 3

    simple_attribute_definition_ = ( attribute_name_ >> "::=" >> static_value_ );
    complex_attribute_definition_ = ( attribute_name_ >> ":=" >> expression_ );

    attribute_definition_ = ( simple_attribute_definition_ | complex_attribute_definition_ );

    attributes_definition_ = ( attribute_definition_ % ',' );

    //-- 4

    events_to_consume_ = (event_name_ % ',');

    //-- Rule

    tesla_rule_ %=    lexeme["assign"]    >> events_declaration_
                >>    lexeme["define"]    >> event_definition_
                >>    lexeme["from"]      >> trigger_pattern_definition_
                >> -( lexeme["where"]     >> attributes_definition_ )
                >> -( lexeme["consuming"] >> events_to_consume_ )
                >> ';';
  }

  qi::rule<It, event_name()> event_name_;
  qi::rule<It, attribute_name()> attribute_name_;
  qi::rule<It, parameter_name()> parameter_name_;

  qi::rule<It, std::string()> strlit_;
  qi::rule<It, static_value()> static_value_;

  qi::rule<It, event_declaration(), ascii::space_type> event_declaration_;
  qi::rule<It, std::vector<event_declaration>(), ascii::space_type> events_declaration_;

  qi::symbols<char, attribute_declaration::attribute_type> type_token_;
  qi::rule<It, attribute_declaration::attribute_type()> attribute_type_;
  qi::rule<It, attribute_declaration(), ascii::space_type> attribute_declaration_;

  qi::rule<It, event_definition(), ascii::space_type> event_definition_;
  
  qi::rule<It, attribute_reference(), ascii::space_type> attribute_reference_;
  
  qi::rule<It, parameter_mapping(), ascii::space_type> parameter_mapping_;
  
  qi::rule<It, parameter_atom(), ascii::space_type> parameter_atom_;

  qi::symbols<char, factor::sign> factor_sign_token_;
  qi::rule<It, factor::sign(), ascii::space_type> factor_sign_;
  qi::symbols<char, term::op_type> term_mul_div_op_type_token_;
  qi::symbols<char, term::op_type> term_add_sub_op_type_token_;
  qi::rule<It, term::op_type(), ascii::space_type> term_mul_div_op_type_;
  qi::rule<It, term::op_type(), ascii::space_type> term_add_sub_op_type_;
  qi::rule<It, expression(), ascii::space_type> expression_;
  qi::rule<It, expression(), ascii::space_type> term_;
  qi::rule<It, atom(), ascii::space_type> factor_;

  qi::symbols<char, simple_attribute_constraint::op_type> constr_op_type_token_;
  qi::rule<It, simple_attribute_constraint::op_type() > constr_op_type_;
  qi::rule<It, simple_attribute_constraint(), ascii::space_type> simple_attribute_constraint_;
  qi::rule<It, complex_attribute_constraint(), ascii::space_type> complex_attribute_constraint_;
  qi::rule<It, attribute_constraint(), ascii::space_type> attribute_constraint_;
  
  qi::symbols<char, aggregate_atom::aggregation_type> agg_type_token_;
  qi::rule<It, aggregate_atom::aggregation_type()> agg_type_;
  qi::rule<It, aggregate_atom(), ascii::space_type> aggregate_atom_;

  qi::symbols<char, within_reference::delta_type> delta_type_token_;
  qi::rule<It, within_reference::delta_type() > delta_type_;
  qi::rule<It, within_reference(), ascii::space_type> within_reference_;
  qi::rule<It, between_reference(), ascii::space_type> between_reference_;
  qi::rule<It, predicate_reference(), ascii::space_type> predicate_reference_;

  qi::symbols<char, positive_predicate::selection_policy> selection_policy_token_;
  qi::rule<It, positive_predicate::selection_policy() > selection_policy_;

  qi::rule<It, predicate_parameter(), ascii::space_type> predicate_parameter_;

  qi::rule<It, predicate(), ascii::space_type> predicate_;
  qi::rule<It, positive_predicate(), ascii::space_type> positive_predicate_;
  qi::rule<It, negative_predicate(), ascii::space_type> negative_predicate_;

  qi::rule<It, trigger_pattern_definition(), ascii::space_type> trigger_pattern_definition_;
  
  qi::rule<It, simple_attribute_definition(), ascii::space_type> simple_attribute_definition_;
  qi::rule<It, complex_attribute_definition(), ascii::space_type> complex_attribute_definition_;
  qi::rule<It, attribute_definition(), ascii::space_type> attribute_definition_;

  qi::rule<It, std::vector<attribute_definition>(), ascii::space_type> attributes_definition_;

  qi::rule<It, std::vector<event_name>(), ascii::space_type> events_to_consume_;

  qi::rule<It, tesla_rule(), ascii::space_type> tesla_rule_;
};

//--------------------------------------------------------------

typedef std::string::const_iterator iter;

bool parse( std::string const& rule_in, tesla_rule& rule_out ){
  iter curr = rule_in.begin();
  iter end = rule_in.end();

  ascii::space_type ws;
  tesla_grammar<iter> gram;
  
  if (phrase_parse(curr, end, gram, ws, rule_out) && curr == end)
    return true;
  
  return false;
}

#include <Parser.hpp>

#include <Packets/RulePkt.h>
#include <Packets/SubPkt.h>

//all these conversions should dissapear

ValType getValType( attribute_declaration::attribute_type const& attr_type ){
  ValType type;
  switch( attr_type ){
    case attribute_declaration::string_type: type = STRING; break;
    case attribute_declaration::int_type: type = INT; break;
    case attribute_declaration::float_type: type = FLOAT; break;
    case attribute_declaration::bool_type: type = BOOL; break;
  }
  return type;
}

CompKind getCompKind( positive_predicate::selection_policy const& sel_pol ){
  CompKind kind;
  switch( sel_pol ){
    case positive_predicate::each_policy: kind = EACH_WITHIN; break;
    case positive_predicate::first_policy: kind = FIRST_WITHIN; break;
    case positive_predicate::last_policy: kind = LAST_WITHIN; break;
  }
  return kind;
}

//comparison !? in java they are named ConstraintOp
Op getOp( simple_attribute_constraint::op_type const& o_type ){
  Op op;
  switch( o_type ){
    case simple_attribute_constraint::eq_op: op = EQ; break;
    case simple_attribute_constraint::gt_op: op = GT; break;
    case simple_attribute_constraint::ge_op: op = GE; break;
    case simple_attribute_constraint::lt_op: op = LT; break;
    case simple_attribute_constraint::le_op: op = LE; break;
    case simple_attribute_constraint::neq_op: op = NE; break;
    default: break;
  }
  return op;
}

OpTreeOperation getOpTreeOperation( term::op_type const& o_type ){
  OpTreeOperation op;
  switch( o_type ){
    case term::mul_op: op = MUL; break;
    case term::div_op: op = DIV; break;
    case term::add_op: op = ADD; break;
    case term::sub_op: op = SUB; break;
    /* AND & OR */
  }
  return op;
}

/*
enum ValRefType {
  RULEPKT = 0,
  STATIC = 1
};
*/

AggregateFun getAggregateFun( aggregate_atom::aggregation_type const& agg_type ){
  AggregateFun fun = NONE;
  switch( agg_type ){
    case aggregate_atom::avg_agg: fun = AVG; break;
    case aggregate_atom::sum_agg: fun = SUM; break;
    case aggregate_atom::min_agg: fun = MIN; break;
    case aggregate_atom::max_agg: fun = MAX; break;
    case aggregate_atom::count_agg: fun = COUNT; break;
  }
  return fun;
}

struct translation_context{
  translation_context() : rule_pkt(new RulePkt( false )), ce_template(NULL) {}

  RulePkt* rule_pkt;
  CompositeEventTemplate* ce_template;

  std::map< event_name, unsigned int > event_ids;
  std::map< attribute_name, attribute_declaration::attribute_type > attr_types;
  std::vector< event_name > predicate_names; //could be an event name or an alias...

  unsigned int get_event_id( event_name const& name ){
    unsigned int id = std::numeric_limits<short>::max(); //just some large value...
    std::map< event_name, unsigned int >::iterator it = event_ids.find( name );
    if( it != event_ids.end() ) id = it->second;
    return id;
  }

  unsigned int get_predicate_index( std::string const& pred ){
    unsigned int idx = std::numeric_limits<short>::max(); //just some large value...
    std::vector<std::string>::iterator it = std::find( predicate_names.begin(), predicate_names.end(), pred );
    if( it != predicate_names.end() ) idx = std::distance(predicate_names.begin(), it);
  }

};

class pattern_predicate_visitor : public boost::static_visitor<>{
  translation_context& ctx;
public:
  pattern_predicate_visitor( translation_context& _ctx ) : ctx(_ctx) {}

  void operator()( positive_predicate& pred ) const{
    TimeMs win;
  //  ctx.rule_pkt->addPredicate(  )
  }

  void operator()( negative_predicate& pred ) const{
  }

};

RulePkt* translate( tesla_rule& rule ){
  translation_context ctx;

  //read the event types
  for( int i=0; i < rule.events_declaration_.size(); ++i )
    ctx.event_ids[ rule.events_declaration_[i].event_name_ ] = rule.events_declaration_[i].id_;
  
  if( rule.event_definition_.attributes_declaration_ ){
    std::vector<attribute_declaration> attributes_declaration = *(rule.event_definition_.attributes_declaration_);
    for( int i=0; i < attributes_declaration.size(); ++i )
      ctx.attr_types[ attributes_declaration[i].attribute_name_ ] = attributes_declaration[i].attribute_type_;
  }
  
  //create the template...
  ctx.ce_template = new CompositeEventTemplate( ctx.event_ids[rule.event_definition_.event_name_] );

  //process the trigger
  predicate root = rule.trigger_pattern_.trigger_predicate_;
  ctx.rule_pkt->addRootPredicate( ctx.event_ids[root.event_name_], NULL, 0 );
 
  //process the rest of predicates
  if( rule.trigger_pattern_.pattern_predicates_ ){
    pattern_predicate_visitor visitor(ctx);
    std::vector<pattern_predicate> pattern_predicates = *(rule.trigger_pattern_.pattern_predicates_);
    for( int i=0; i < pattern_predicates.size(); ++i )
      boost::apply_visitor( visitor, pattern_predicates[i] );
  }
  
  //add predicates first
  if( rule.events_to_consume_ ){
    std::vector<event_name> events_to_consume = *(rule.events_to_consume_);
    for( int i=0; i < events_to_consume.size(); ++i )
      ctx.rule_pkt->addConsuming( ctx.event_ids[ events_to_consume[i] ] ); //!?!?
  }

  ctx.rule_pkt->setCompositeEventTemplate(ctx.ce_template);
  return ctx.rule_pkt;
}

RulePkt* buildRule( std::string const& rule_in ){
  tesla_rule rule_out;
  if(parse( rule_in, rule_out ))
    return translate( rule_out );
  return NULL;
}

SubPkt* buildPlainSubscription( RulePkt* rule ){
  return NULL;
}
