#include <boost/spirit/include/qi.hpp>
#include <boost/fusion/adapted.hpp>
#include <boost/optional.hpp>
#include <boost/variant.hpp>

#include <iostream>
#include <string>
#include <vector>
#include <stack>

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

#include <Parser.hpp>

#include <Packets/RulePkt.h>
#include <Packets/SubPkt.h>

//all these conversions should dissapear

#define GUARD( CALL ) GUARD_EXT( CALL, "error" )
#define GUARD_EXT( CALL, ERR ) do{if(!CALL){ std::cout << ERR << "\n"; }}while(0)

ValType get_val_type( attribute_declaration::attribute_type attr_type ){
  ValType type;
  switch( attr_type ){
    case attribute_declaration::string_type: type = STRING; break;
    case attribute_declaration::int_type: type = INT; break;
    case attribute_declaration::float_type: type = FLOAT; break;
    case attribute_declaration::bool_type: type = BOOL; break;
  }
  return type;
}

CompKind get_comp_kind( positive_predicate::selection_policy sel_pol ){
  CompKind kind;
  switch( sel_pol ){
    case positive_predicate::each_policy: kind = EACH_WITHIN; break;
    case positive_predicate::first_policy: kind = FIRST_WITHIN; break;
    case positive_predicate::last_policy: kind = LAST_WITHIN; break;
  }
  return kind;
}

//comparison !? in java they are named ConstraintOp
Op get_op( simple_attribute_constraint::op_type o_type ){
  Op op;
  switch( o_type ){
    case simple_attribute_constraint::eq_op: op = EQ; break;
    case simple_attribute_constraint::gt_op: op = GT; break;
    case simple_attribute_constraint::ge_op: op = GE; break;
    case simple_attribute_constraint::lt_op: op = LT; break;
    case simple_attribute_constraint::le_op: op = LE; break;
    case simple_attribute_constraint::neq_op: op = NE; break;
    default: break; //!!??
  }
  return op;
}

OpTreeOperation get_op_tree_operation( term::op_type o_type ){
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

AggregateFun get_aggregate_fun( aggregate_atom::aggregation_type agg_type ){
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

unsigned int delta_type( within_reference::delta_type d_type ){
  unsigned int ms = 1;
  switch( d_type ){
    case within_reference::millisecs: break;
    case within_reference::secs: ms = 1000; break;
    case within_reference::mins: ms = 60*1000; break;
    case within_reference::hours: ms = 60*60*1000; break;
    case within_reference::days: ms = 24*60*60*1000; break;
  }
  return ms;
};

//-- print expressions in postfix format...

std::ostream& operator<<( std::ostream& os, attribute_declaration::attribute_type attr_type ){
  switch( attr_type ){
    case attribute_declaration::string_type: os << "STRING"; break;
    case attribute_declaration::int_type: os << "INT"; break;
    case attribute_declaration::float_type: os << "FLOAT"; break;
    case attribute_declaration::bool_type: os << "BOOL"; break;
  }
  return os;
}

std::ostream& operator<<(std::ostream& os, simple_attribute_constraint::op_type o_type){
  switch( o_type ){
    case simple_attribute_constraint::eq_op: os << "=="; break;
    case simple_attribute_constraint::gt_op: os << ">"; break;
    case simple_attribute_constraint::ge_op: os << ">="; break;
    case simple_attribute_constraint::lt_op: os << "<"; break;
    case simple_attribute_constraint::le_op: os << "<="; break;
    case simple_attribute_constraint::neq_op: os << "!="; break;
    default: break; //!!??
  }
  return os;
}

std::ostream& operator<<(std::ostream& os, attribute_reference const& ref){
  return os << ref.event_name_ << "." << ref.attribute_name_;
}

class parameter_atom_visitor : public boost::static_visitor<>{
  std::ostream& s;
public:
  parameter_atom_visitor( std::ostream& _s ) : s(_s) {}

  void operator() ( attribute_reference const& attr_ref ){ s << " " << attr_ref << " "; }
  void operator() ( parameter_name const& pn ){ s << " " << pn << " "; }
  void operator() ( static_value const& sv ){ s << " " << sv << " "; }
};

std::ostream& operator<<(std::ostream& os, parameter_atom const& atm){
  parameter_atom_visitor v(os);
  boost::apply_visitor( v, atm );
  return os;
}

std::ostream& operator<<(std::ostream& os, aggregate_atom const& agg){
  return os << " xxx ";
}

std::ostream& operator<<(std::ostream& os, factor const& fct){
  return os << fct.atom_ << ( fct.sign_ == factor::neg_sgn ? " - " : "" );
}

std::ostream& operator<<(std::ostream& os, term::op_type const& op){
  switch( op ){
    case term::mul_op: os << " * "; break;
    case term::div_op: os << " / "; break;
    case term::add_op: os << " + "; break;
    case term::sub_op: os << " - "; break;
  }
  return os;
}

std::ostream& operator<<(std::ostream& os, term const& trm){
  return os << trm.atom_ << trm.op_;
}

std::ostream& operator<<(std::ostream& os, expression const& expr){
  os << expr.first_;
  for( int i=0; i<expr.rest_.size(); ++i )
    os << expr.rest_[i];
  return os;
}

//--

struct translation_context{
  translation_context() : rule_pkt(new RulePkt( false )), ce_template(NULL) {}

  RulePkt* rule_pkt;
  CompositeEventTemplate* ce_template;
  
  struct parameter{
    unsigned int ev_id;   //the index of a predicate ...
    attribute_name attr_name;
  };
  
  enum predicate_type{ root_predicate, positive_predicate, negative_predicate };
  enum reference_type{ within_ref, between_ref };

  struct predicate_context{
    predicate_context() : pred_type( root_predicate ) {}

    event_name predicate_name;  //used by the parameter mappings & agg (pos&neg)
    predicate_type pred_type;   
    reference_type ref_type;    //useful only for positive,negative predicates (not root)
    CompKind sel_policy;        //useful only for positive preducates
    unsigned int refers_to_1;   //useful for positive&negative predicates (not for root)
    unsigned int refers_to_2;   //useful only for negative predicates using a between predicate reference
    TimeMs win_in_millisecs;    //useful for positive & pos agg predicates and negative predicates using within predicate reference

    std::vector< Constraint > constraints;
  };

  std::map< event_name, unsigned int > event_ids;
  std::map< attribute_name, attribute_declaration::attribute_type > attr_types;
  std::map< parameter_name, parameter > parameters;
  std::vector< event_name > predicate_names; //could be an event name or an alias... and it seems to be used for indexing
  
  unsigned int get_event_id( event_name const& name ){
    unsigned int id = std::numeric_limits<short>::max(); //just some large value...
    std::map< event_name, unsigned int >::iterator it = event_ids.find( name );
    if( it != event_ids.end() ) id = it->second;
    return id;
  }
  
  unsigned int get_predicate_index( event_name const& pred ){
    unsigned int idx = std::numeric_limits<short>::max(); //just some large value...
    std::vector<std::string>::iterator it = std::find( predicate_names.begin(), predicate_names.end(), pred );
    if( it != predicate_names.end() ) idx = std::distance(predicate_names.begin(), it);
    return idx;
  }

  void visit_predicate( predicate const& pred, predicate_context& pred_ctx ){
    unsigned int event_id = get_event_id(pred.event_name_);

    if( pred.alias_ ){
      pred_ctx.predicate_name = *pred.alias_;            //use the alias if available
      event_ids[ pred_ctx.predicate_name ] = event_id;   //add also the alias to the name-id map
    }else{
      pred_ctx.predicate_name = pred.event_name_;        //otherwise use the name
    }

    switch( pred_ctx.pred_type ){
      case translation_context::root_predicate:
        GUARD( rule_pkt->addRootPredicate( event_id, &pred_ctx.constraints[0], pred_ctx.constraints.size() ) );
      break;
      case translation_context::positive_predicate:{
        GUARD( rule_pkt->addPredicate( event_id, &pred_ctx.constraints[0], pred_ctx.constraints.size(), pred_ctx.refers_to_1, pred_ctx.win_in_millisecs, pred_ctx.sel_policy ) );
      }break;
      case translation_context::negative_predicate:{
        if( pred_ctx.ref_type == within_ref ){
          GUARD( rule_pkt->addTimeBasedNegation( event_id, &pred_ctx.constraints[0], pred_ctx.constraints.size(), pred_ctx.refers_to_1, pred_ctx.win_in_millisecs ) );
        }else{//between
          GUARD( rule_pkt->addNegationBetweenStates( event_id, &pred_ctx.constraints[0], pred_ctx.constraints.size(), pred_ctx.refers_to_1, pred_ctx.refers_to_2 ) );
        }
      }break;
    };

    if( pred_ctx.pred_type != translation_context::negative_predicate ){
      predicate_names.push_back(pred_ctx.predicate_name);
    }
  }

  void visit_within_reference( predicate_context& pred_ctx, within_reference const& ref ){
    pred_ctx.ref_type = within_ref;
    pred_ctx.refers_to_1 = get_predicate_index( ref.event_name_ );
    pred_ctx.win_in_millisecs = ref.delta_ * delta_type( ref.delta_type_ );//transform in millisconds
  }

  void visit_between_reference( predicate_context& pred_ctx, between_reference const& ref ){
    pred_ctx.ref_type = between_ref;
    pred_ctx.refers_to_1 = get_predicate_index( ref.fst_event_name_ );
    pred_ctx.refers_to_2 = get_predicate_index( ref.snd_event_name_ );
  }
};

//used by simple_attribute_constraint and simple_attribute_definition...
template<typename T>
class static_value_visitor : public boost::static_visitor<>{
  T& v;
public:
  static_value_visitor( T& _v ) : v(_v) {}

  void operator() ( std::string const& val ){
    memset(  v.stringVal, 0, NAME_LEN+1 );
    strncpy( v.stringVal, val.c_str(), NAME_LEN );
    v.type = STRING;
  }
  void operator() ( int val ){ v.intVal = val; v.type = INT; }
  void operator() ( float val ){ v.floatVal = val; v.type = FLOAT; }
  void operator() ( bool val ){ v.boolVal = val; v.type = BOOL; }
};

//used by complex_attribute_constraint and complex_attribute_definition...
class expression_processor : public boost::static_visitor<>{
  translation_context& ctx;
  expression& expr;
  ValType type;
  std::stack<OpTree*> nodes;

  struct static_value_processor : public boost::static_visitor<>{
    static_value_processor( expression_processor& ep ) : parent(ep) {}
    
    void operator() ( std::string const& val){
      StaticValueReference* ref = new StaticValueReference( const_cast<char*>(val.c_str()) );
      parent.nodes.push( new OpTree( ref, STRING ) );
    }

    void operator() ( int const& val ){ 
      StaticValueReference* ref = new StaticValueReference( val );
      parent.nodes.push( new OpTree( ref, INT ) );
    }

    void operator() ( float const& val ){
      StaticValueReference* ref = new StaticValueReference( val );
      parent.nodes.push( new OpTree( ref, FLOAT ) );
    }

    void operator() ( bool const& val ){
      StaticValueReference* ref = new StaticValueReference( val );
      parent.nodes.push( new OpTree( ref, BOOL ) );
    }
    
    expression_processor& parent;
  };

  struct parameter_atom_processor : public boost::static_visitor<>{
    parameter_atom_processor( expression_processor& ep ) : parent(ep) {}

    void operator() ( attribute_reference const& atr ){
      unsigned int idx = parent.ctx.get_predicate_index( atr.event_name_ );
      RulePktValueReference* ref = new RulePktValueReference( idx, const_cast<char*>(atr.attribute_name_.c_str()), STATE );
      parent.nodes.push( new OpTree( ref, parent.type ) );
    }
    
    void operator() ( parameter_name const& pn ){
      translation_context::parameter& param = parent.ctx.parameters[ pn ];
      RulePktValueReference* ref = new RulePktValueReference( param.ev_id, const_cast<char*>(param.attr_name.c_str()), STATE );
      parent.nodes.push( new OpTree( ref, parent.type ) );
    }

    void operator() ( static_value const& sv ){
      static_value_processor svp( parent );
      boost::apply_visitor( svp, sv );
    }

    expression_processor& parent;
  };
 
  OpTree* pop_last(){
    OpTree *node = nodes.top();
    nodes.pop();
    return node;
  }

  void create_inner_node( term::op_type o_type ){
    OpTree* right = pop_last();
    OpTree* left = pop_last();
    nodes.push( new OpTree(left, right, get_op_tree_operation(o_type), type) );
  }

public:
  expression_processor(translation_context& _ctx, expression& _expr, attribute_declaration::attribute_type _type ) 
    : ctx(_ctx), expr(_expr), type(get_val_type(_type)) {}
  
  void operator() ( atom const& atm ){ boost::apply_visitor( *this, atm ); }

  void operator() ( parameter_atom const& pa ){
    parameter_atom_processor pap(*this);
    boost::apply_visitor( pap, pa );
  }

  void operator() ( aggregate_atom const& atm ){
    
  }
  
  //here we should define some behaviour for bool and string too...
  void operator() ( factor const& fct ){
    StaticValueReference* ref = NULL;

    if( fct.sign_ == factor::neg_sgn ){
      /* add a zero node */
      switch(type){
        case INT: ref = new StaticValueReference( (int)0 ); break;
        case FLOAT: ref = new StaticValueReference( (float)0 ); break;
        default: break;
      }
      if(erf)
        nodes.push( new OpTree( ref, type ) );
    }

    boost::apply_visitor( *this, fct.atom_ );

    if( fct.sign_ == factor::neg_sgn && ref){
      /* pop the last 2 nodes and create an inner node */
      create_inner_node( term::sub_op );
    }
  }
  
  void operator() ( term const& trm ){
    boost::apply_visitor( *this, trm.atom_ );
    /* now pop last 2 nodes and create an inner node */
    create_inner_node( trm.op_ );
  }
  
  void operator() ( expression const& exp ){
    boost::apply_visitor( *this, exp.first_ );
    BOOST_FOREACH(term const& trm, exp.rest_) (*this)(trm);
  }

  OpTree* process(){
    (*this)( expr );
    return pop_last();
  }
};

class predicate_parameter_visitor : public boost::static_visitor<>{
  translation_context& ctx;
  translation_context::predicate_context& pred_ctx;
  bool simple;

public:
  predicate_parameter_visitor( translation_context& _ctx, translation_context::predicate_context& _pred_ctx )
    : ctx(_ctx), pred_ctx(_pred_ctx), simple(true) {}

  void toggle(){ simple = !simple; }

  void operator() ( parameter_mapping const& mp ){
    if( !simple ){
      translation_context::parameter param;
      param.ev_id = ctx.get_predicate_index( pred_ctx.predicate_name );
      param.attr_name = mp.attribute_name_;
      ctx.parameters[ mp.parameter_name_ ] = param;
    }
  }

  void operator() ( simple_attribute_constraint const& sac ){
    if( simple ){
      Constraint constraint;
      memset(  constraint.name, 0, NAME_LEN+1 );
      strncpy( constraint.name, sac.attribute_name_.c_str(), NAME_LEN );
      constraint.op = get_op( sac.op_ );
      static_value_visitor<Constraint> val_visitor( constraint );
      boost::apply_visitor( val_visitor, sac.static_value_ );
      pred_ctx.constraints.push_back( constraint );
    }
  }
  
  void operator() ( complex_attribute_constraint const& cac ){
    if( !simple ){
      StateType pred_type = (pred_ctx.pred_type == translation_context::negative_predicate ? NEG : STATE);

      //std::cout << (pred_type == STATE ? "pos":"neg") << std::endl;
      //std::cout << "AttrConstr Expr: " << cac.expression_ << "\n";

      expression_processor ep( ctx, const_cast<expression&>(cac.expression_), cac.attribute_type_ );
      OpTree *right = ep.process();
      
      ValType type = get_val_type( cac.attribute_type_ );
      unsigned int idx = ctx.get_predicate_index( pred_ctx.predicate_name );
      RulePktValueReference *ref = new RulePktValueReference( idx, const_cast<char*>(cac.attribute_name_.c_str()), pred_type );
      OpTree *left = new OpTree( ref, type );

      //std::cout << "type: " << cac.attribute_type_ << " idx: " << idx << " attr: " << cac.attribute_name_ << " op: " << cac.op_ << "\n";

      Op op = get_op( cac.op_ );
      if( pred_type == NEG ){
        GUARD(ctx.rule_pkt->addComplexParameterForNegation( op, type, left, right )); //negative
      }else{
        GUARD(ctx.rule_pkt->addComplexParameter( op, type, left, right ));            //root & positive
      }
    }
  }
};

class predicate_reference_visitor : public boost::static_visitor<>{
  translation_context& ctx;
  translation_context::predicate_context& pred_ctx;
public:
  predicate_reference_visitor( translation_context& _ctx, translation_context::predicate_context& _pred_ctx )
    : ctx(_ctx), pred_ctx(_pred_ctx) {}

  void operator() ( within_reference const& ref ){ ctx.visit_within_reference( pred_ctx, ref ); }
  void operator() ( between_reference const& ref ){ ctx.visit_between_reference( pred_ctx, ref ); }
};

class pattern_predicate_visitor : public boost::static_visitor<>{
  translation_context& ctx;
  translation_context::predicate_context pred_ctx;

  static void visit_parameters( boost::optional<std::vector<predicate_parameter> >& params, predicate_parameter_visitor& visitor ){
    if( params ){
      std::vector<predicate_parameter>& vp = *params;
      for( int i=0; i< vp.size(); ++i ) boost::apply_visitor( visitor, vp[i] );
    }
  }

public:
  pattern_predicate_visitor( translation_context& _ctx ) : ctx(_ctx) {}

  void visit( predicate& pred ){
    predicate_parameter_visitor param_visitor( ctx, pred_ctx );
    visit_parameters( pred.predicate_parameters_, param_visitor );  //read simple constraints ( attr_constraint )
    ctx.visit_predicate( pred, pred_ctx );                          //create the predicate...
    param_visitor.toggle();                                        
    visit_parameters( pred.predicate_parameters_, param_visitor );  //read complex constraints ( param_mapping & attr_parameters )
  }

  void operator()( positive_predicate& pred ) {
    pred_ctx.pred_type = translation_context::positive_predicate;
    pred_ctx.sel_policy = get_comp_kind( pred.selection_policy_ );
    ctx.visit_within_reference( pred_ctx, pred.reference_ );
    visit( pred.predicate_ );
  }

  void operator()( negative_predicate& pred ) {
    pred_ctx.pred_type = translation_context::negative_predicate;
    predicate_reference_visitor ref_visitor( ctx, pred_ctx );
    boost::apply_visitor( ref_visitor, pred.reference_ );
    visit( pred.predicate_ );
  }
};

class attribute_visitor : public boost::static_visitor<>{
  translation_context& ctx;
public:
  attribute_visitor( translation_context& _ctx ) : ctx(_ctx) {}

  void operator()( simple_attribute_definition& attr ) {
    Attribute attribute;
    memset(  attribute.name, 0, NAME_LEN+1 );
    strncpy( attribute.name, attr.attribute_name_.c_str(), NAME_LEN );
    static_value_visitor<Attribute> val_visitor( attribute );
    boost::apply_visitor( val_visitor, attr.static_value_ );
    ctx.ce_template->addStaticAttribute( attribute );
  }

  void operator()( complex_attribute_definition& attr ) {
    //std::cout << "\nAttrDef Expr: " << attr.expression_ << "\n";
    expression_processor ep( ctx, attr.expression_, ctx.attr_types[ attr.attribute_name_ ] );
    ctx.ce_template->addAttribute(const_cast<char*>(attr.attribute_name_.c_str()), ep.process());
  }
};

//--

// I have to say that this RulePkt class would greatly benefit from some sort of refactoring/cleaning...
RulePkt* translate( tesla_rule& rule ){
  translation_context ctx;

  //read the event types
  for( int i=0; i < rule.events_declaration_.size(); ++i )
    ctx.event_ids[ rule.events_declaration_[i].event_name_ ] = rule.events_declaration_[i].id_;
  
  if( rule.event_definition_.attributes_declaration_ ){
    std::vector<attribute_declaration>& attributes_declaration = *(rule.event_definition_.attributes_declaration_);
    for( int i=0; i < attributes_declaration.size(); ++i )
      ctx.attr_types[ attributes_declaration[i].attribute_name_ ] = attributes_declaration[i].attribute_type_;
  }
  
  //create the template...
  ctx.ce_template = new CompositeEventTemplate( ctx.get_event_id(rule.event_definition_.event_name_) );

  {//process the trigger(root) predicate
    pattern_predicate_visitor root_visitor(ctx);
    root_visitor.visit( rule.trigger_pattern_.trigger_predicate_ );
  }
 
  //process the rest of predicates
  if( rule.trigger_pattern_.pattern_predicates_ ){
    std::vector<pattern_predicate>& pattern_predicates = *(rule.trigger_pattern_.pattern_predicates_);
    for( int i=0; i < pattern_predicates.size(); ++i ){
      pattern_predicate_visitor pred_visitor(ctx);
      boost::apply_visitor( pred_visitor, pattern_predicates[i] );
    }
  }

  //process the definitions
  if( rule.attributes_definition_ ){
    std::vector<attribute_definition>& attributes_definition = *( rule.attributes_definition_ );
    attribute_visitor attr_visitor(ctx);
    for( int i=0; i < attributes_definition.size(); ++i ){
      boost::apply_visitor( attr_visitor, attributes_definition[i] );
    }
  }
  
  //add predicates first... in order to be able to use their indexes... !?!?
  if( rule.events_to_consume_ ){
    std::vector<event_name> events_to_consume = *(rule.events_to_consume_);
    for( int i=0; i < events_to_consume.size(); ++i )
      GUARD( ctx.rule_pkt->addConsuming( ctx.get_predicate_index( events_to_consume[i] ) ) );
  }

  ctx.rule_pkt->setCompositeEventTemplate(ctx.ce_template);
  return ctx.rule_pkt;
}

//--

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

RulePkt* buildRule( std::string const& rule_in ){
  tesla_rule rule_out;
  if(parse( rule_in, rule_out ))
    return translate( rule_out );
  return NULL;
}

//--

SubPkt* buildPlainSubscription( RulePkt* rule ){
	return new SubPkt(rule->getCompositeEventTemplate()->getEventType(), NULL, 0);
}
