/*
 * Copyright (C) 2011 Francesco Feltrinelli <first_name DOT last_name AT gmail DOT com>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include "external.hpp"
#include "server.hpp"
#include "test.hpp"
#include "util.hpp"

#include "Parser.hpp"

using concept::server::SOEPServer;
using namespace concept::test;
using concept::util::Logging;
using namespace std;


void runServer(bool useGPU){
	// Create server with default port and #threads = #CPUs
	SOEPServer server(SOEPServer::DEFAULT_PORT, boost::thread::hardware_concurrency(), false, useGPU);

  //install rules
  /*
  RulePkt *rule = buildRule( "...some rule..." );
  server.getEngine().processRulePkt(rule);
  */

	server.run();
}

void testEngine(){
	TRexEngine engine(2);
	RuleR1 testRule;

	engine.processRulePkt(testRule.buildRule());
	engine.finalize();

	ResultListener* listener= new TestResultListener(testRule.buildSubscription());
	engine.addResultListener(listener);

	vector<PubPkt*> pubPkts= testRule.buildPublication();
	for (vector<PubPkt*>::iterator it= pubPkts.begin(); it != pubPkts.end(); it++){
		engine.processPubPkt(*it);
	}
	/* Expected output: complex event should be created by T-Rex and published
	 * to the TestResultListener, which should print it to screen.
	 */
}

// To come up with a clever way to write these tests... !!!
/* 
Maybe some sort of framework where for each test you will need to provide:
- a first file defining the rules to test
- a second file specifing the events to generate
- a third file specifing the expected events
And after running a test the framework will compare generated with expected events...
*/

void testParser1(){
  std::cout << "\n---Test1---\n" << std::flush;

  TRexEngine *engine = new TRexEngine(2);
  engine->finalize();

  RulePkt *rule = buildRule( "assign 1=>Ev1, 2=>Ev2, 3=>Ev3 define Ev1( val11: int ) from Ev2( val21 => $a ) and last Ev3( [int] val31 > $a ) within 10 mins from Ev2 where val11 := -(2*$a-1);");
  if( !rule ){ std::cout << "parsing or translation failed" << std::flush; return; }
  ResultListener* listener= new TestResultListener(buildPlainSubscription( rule ));

  engine->processRulePkt(rule);
  engine->addResultListener(listener);	
  
  //send some data

  {
  Attribute attr31[1];
	strcpy(attr31[0].name, "val31");
	attr31[0].type= INT;
	attr31[0].intVal= 20;
  PubPkt* pubPkt1 = new PubPkt(3, attr31, 1);
  engine->processPubPkt(pubPkt1);
  boost::this_thread::sleep(boost::posix_time::milliseconds(10)); //!?
  }

  {
  Attribute attr21[1];
	strcpy(attr21[0].name, "val21");
	attr21[0].type= INT;
	attr21[0].intVal = 19;
	PubPkt* pubPkt2 = new PubPkt(2, attr21, 1);
  engine->processPubPkt(pubPkt2);
  boost::this_thread::sleep(boost::posix_time::milliseconds(10)); //!?
  }
  
  {
  Attribute attr21[1];
	strcpy(attr21[0].name, "val21");
	attr21[0].type= INT;
	attr21[0].intVal = 18;
	PubPkt* pubPkt2 = new PubPkt(2, attr21, 1);
  engine->processPubPkt(pubPkt2);
  boost::this_thread::sleep(boost::posix_time::milliseconds(10)); //!?
  }

  delete engine;
  delete listener;
}

void testParser2(){
  std::cout << "\n---Test2---\n" << std::flush;

  TRexEngine *engine = new TRexEngine(2);
  engine->finalize();

  RulePkt *rule = buildRule( "assign 1=>Ev1, 2=>Ev2, 3=>Ev3 define Ev1( val11: int ) from Ev2( val21 => $a ) and last Ev3( [int] val31 > $a ) within 10 mins from Ev2 where val11 := $a;");
  if( !rule ){ std::cout << "parsing or translation failed" << std::flush; return; }
  ResultListener* listener= new TestResultListener(buildPlainSubscription( rule ));

  engine->processRulePkt(rule);
  engine->addResultListener(listener);	
  
  //send some data

  {
  Attribute attr31[1];
	strcpy(attr31[0].name, "val31");
	attr31[0].type= INT;
	attr31[0].intVal= 20;
  PubPkt* pubPkt1 = new PubPkt(3, attr31, 1);
  engine->processPubPkt(pubPkt1);
  boost::this_thread::sleep(boost::posix_time::milliseconds(10)); //!?

  Attribute attr21[1];
	strcpy(attr21[0].name, "val21");
	attr21[0].type= INT;
	attr21[0].intVal = 19;
	PubPkt* pubPkt2 = new PubPkt(2, attr21, 1);
  engine->processPubPkt(pubPkt2);
  boost::this_thread::sleep(boost::posix_time::milliseconds(10)); //!?
  }

  delete engine;
  delete listener;
}

void testParser3(){
  std::cout << "\n---Test3---\n" << std::flush;

  TRexEngine *engine = new TRexEngine(2);
  engine->finalize();

  RulePkt *rule = buildRule( "assign 1=>Ev1, 2=>Ev2, 3=>Ev3 define Ev1( val11: int ) from Ev2( val21 => $a ) and last Ev3( [int] val31 > $a ) within 10 mins from Ev2 where val11 := MIN( Ev2.val21() ) within 10 mins from Ev2;");
  if( !rule ){ std::cout << "parsing or translation failed" << std::flush; return; }
  ResultListener* listener= new TestResultListener(buildPlainSubscription( rule ));

  engine->processRulePkt(rule);
  engine->addResultListener(listener);	
  
  //send some data

  {
  Attribute attr31[1];
	strcpy(attr31[0].name, "val31");
	attr31[0].type= INT;
	attr31[0].intVal= 30;
  PubPkt* pubPkt1 = new PubPkt(3, attr31, 1);
  engine->processPubPkt(pubPkt1);
  boost::this_thread::sleep(boost::posix_time::milliseconds(10)); //!?
  }

  {
  Attribute attr21[1];
	strcpy(attr21[0].name, "val21");
	attr21[0].type= INT;
	attr21[0].intVal = 19;
	PubPkt* pubPkt2 = new PubPkt(2, attr21, 1);
  engine->processPubPkt(pubPkt2);
  boost::this_thread::sleep(boost::posix_time::milliseconds(10)); //!?
  }
  
  {
  Attribute attr21[1];
	strcpy(attr21[0].name, "val21");
	attr21[0].type= INT;
	attr21[0].intVal = 18;
	PubPkt* pubPkt2 = new PubPkt(2, attr21, 1);
  engine->processPubPkt(pubPkt2);
  boost::this_thread::sleep(boost::posix_time::milliseconds(10)); //!?
  }

  {
  Attribute attr21[1];
	strcpy(attr21[0].name, "val21");
	attr21[0].type= INT;
	attr21[0].intVal = 20;
	PubPkt* pubPkt2 = new PubPkt(2, attr21, 1);
  engine->processPubPkt(pubPkt2);
  boost::this_thread::sleep(boost::posix_time::milliseconds(10)); //!?
  }

  delete engine;
  delete listener;
}

int main(int argc, char* argv[]){
	Logging::init();

  if (argc==3 && strcmp(argv[1], "-cpp_parser")==0) {
    int test = atoi( argv[2] );
    switch( test ){
    case 1: testParser1(); break;
    case 2: testParser2(); break;
    case 3: testParser3(); break;
    }
    std::cout << "\nBye... :-) \n";
    return 0;
  }

#ifdef HAVE_GTREX
	if (argc==2 && strcmp(argv[1], "-gpu")==0) {
	  cout << "Using GPU engine - GTREX" << endl;
	  runServer(true);
	}
	else {
	  cout << "Using CPU engine - TREX" << endl;
	  runServer(false);
	}
#else
	runServer(false);
#endif
}
