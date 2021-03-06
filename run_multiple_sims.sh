#!/bin/bash


n_flag='false'
s_flag='false'

print_usage() {
  printf "Flag not recognized, please use -n to simulate various noise levels or -s to simulate various sample sizes.\n"
}

while getopts 'ns' flag; do
  case "${flag}" in
    n) n_flag='true' ;;
    s) s_flag='true' ;;
    *) print_usage
       exit 1 ;;
  esac
done

if [ "$n_flag" == "true" ]; then
  for seed in 11 #13 111 131
  do

    sed -i "s/seed = 131/seed = $seed/g" conf/conf.py
    echo "seed $seed"

    python3 simulate_network.py -vqte --network ieee123center

    for i in 0.1 #0.2 0.5 1 2 5 10
    do
      echo "Test with noise level $i"

      sed -i "s/noise_level = 1/noise_level = $i/g" conf/conf.py

      python3 simulate_network.py -qtde --network ieee123center
      python3 identify_network.py -q --max-iterations 20 --network ieee123center --phases 012
      python3 plot_results.py -v --network ieee123center

      sed -i "s/noise_level = $i/noise_level = 1/g" conf/conf.py
    done

    sed -i "s/seed = $seed/seed = 131/g" conf/conf.py
  done
fi

if [ "$s_flag" == "true" ]; then
  for seed in  11 13 111 131
  do

    sed -i "s/seed = 131/seed = $seed/g" conf/conf.py
    echo "seed $seed"

    for i in 1 2 3 5 7 10 15 21 30 45 60
    do
      echo "Test with $i day sample size"

      sed -i "s/days = 7/days = $i/g" conf/simulation.py

      python3 simulate_network.py -qte --network ieee123center
      python3 identify_network.py -q --max-iterations 20 --network ieee123center --phases 012
      python3 plot_results.py -v --network ieee123center

      sed -i "s/days = $i/days = 7/g" conf/simulation.py
    done

    sed -i "s/seed = $seed/seed = 131/g" conf/conf.py
  done
fi

#for seed in 11 13 111 131
#do
#
#  sed -i "s/seed = 131/seed = $seed/g" conf/conf.py
#  echo "seed $seed"
#
#  for i in 1 2 3 5 7 10 15 21 30 45 60
#  do
#    echo "Test with $i day sample size"
#
#    sed -i "s/days = 7/days = $i/g" conf/simulation.py
#
#    python3 simulate_network.py -e
#    python3 identify_network.py -su --max-iterations 0
#
#    sed -i "s/days = $i/days = 7/g" conf/simulation.py
#  done
#
#  sed -i "s/seed = $seed/seed = 131/g" conf/conf.py
#done
