#!/bin/bash



for semente in 960703545 1277478588 1936856304 186872697 1859168769 1598189534 1822174485 1871883252 694388766 188312339 773370613 2125204119 2041095833 1384311643 1000004583 358485174 1695858027 762772169 437720306 939612284
 do
   for gamm in 0.1 0.25 0.5 0.9 0.99
    do
      param="gamma"
      echo $semente

      type_e="deterministic"
      reward="unit"
      python3 main.py --seed="${semente}" --value="${gamm}" --testing_agent_param_name="${param}" --type_env="${type_e}" --rewards="${reward}" &
      sleep 5

      type_e="deterministic"
      reward="weighted"
      echo $semente
      python3 main.py --seed="${semente}" --value="${gamm}" --testing_agent_param_name="${param}" --type_env="${type_e}" --rewards="${reward}" &
      sleep 5

      type_e="stochastic"
      reward="unit"
      python3 main.py --seed="${semente}" --value="${gamm}" --testing_agent_param_name="${param}" --type_env="${type_e}" --rewards="${reward}" &
      sleep 5

      type_e="stochastic"
      reward="weighted"
      python3 main.py --seed="${semente}" --value="${gamm}" --testing_agent_param_name="${param}" --type_env="${type_e}" --rewards="${reward}" &
      sleep 5
      
      
	if [ $gamm = 0.1 ];
	then
	    min_epsilon=0.05
	elif [ $gamm = 0.25 ];
	then
	     min_epsilon=0.1
	elif [ $gamm = 0.5 ];
	then
	     min_epsilon=0.3
	elif [ $gamm = 0.99 ];
	then
	     min_epsilon=0.7		
	elif [ $gamm = 0.9 ];
	then
	    min_epsilon=0.5
	fi
      
      param="min_epsilon"
      echo $semente
      type_e="deterministic"
      reward="unit"
      python3 main.py --seed="${semente}" --value="${min_epsilon}" --testing_agent_param_name="${param}" --type_env="${type_e}" --rewards="${reward}" &
      sleep 5

      type_e="deterministic"
      reward="weighted"
      python3 main.py --seed="${semente}" --value="${min_epsilon}" --testing_agent_param_name="${param}" --type_env="${type_e}" --rewards="${reward}" &
      sleep 5

      type_e="stochastic"
      reward="unit"
      python3 main.py --seed="${semente}" --value="${min_epsilon}" --testing_agent_param_name="${param}" --type_env="${type_e}" --rewards="${reward}" &
      sleep 5

      type_e="stochastic"
      reward="weighted"
      python3 main.py --seed="${semente}" --value="${min_epsilon}" --testing_agent_param_name="${param}" --type_env="${type_e}" --rewards="${reward}" 
    done
  done


