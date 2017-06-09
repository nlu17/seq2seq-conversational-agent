#!/bin/bash

echo "Number of arguments: $#"

if [ $# = 0 ]
then
	echo "Please pass at least one argument (the test source path)."
	exit
fi

test_path=$1
load_model="baseline"

if [ $# = 2 ]
then
	load_model=$2
fi

echo "Loading model: $load_model"

echo "Creating test_source.txt and test_target.txt files. This may take a minute.."

#rm -f data/test_source.txt
#rm -f data/test_target.txt

#creates test_source.txt, test_target.txt
#python3 create_test_datasets.py $test_path

echo "Created test_source.txt and test_target.txt !"



output_file=""
if [ $load_model = "baseline" ]
then 
	echo "Generating baseline results:"

	#This script generates only one file with lines of form 
	# <sentence> <bleu score> <perplexity>
	python3 sample.py --with_attention=False --output_file=ceva_baseline.out --custom_decoder=default --checkpoint_dir=data/baseline_stored/ --ckpt_file=data/baseline_stored/chatbot.ckpt-1065000

	#with this script we use the file from above (ceva_baseline.out) to generate the final output 
	output_file="ceva_baseline.out"
elif [ $load_model = "attention" ]
then
	echo "Generating 4-layered attention model resuts.."
	python3 sample.py --with_attention=True --output_file=ceva_4layer_attention.out --custom_decoder=default --checkpoint_dir=data/attention_stored/ --ckpt_file=data/attention_stored/chatbot.ckpt-829000
	output_file="ceva_4layer_attention.out"
elif [ $load_model = "diversification" ]
then
	echo "Generating results with a 4-layered attention model and a MMI diversification on top of it.."
	output_file="divers"
	#python3 sample.py --with_attention=True --output_file=ceva_4layer_attention.out --custom_decoder=mmi --checkpoint_dir= --ckpt_file=
elif [ $load_model = "beamsearch" ]
then
	echo "Generating results with a 4-layered attention model and a MMI diversification on top of it.."
	output_file="beamsearch"
	#python3 sample.py --with_attention=True --output_file=ceva_4layer_attention.out --custom_decoder=beam --checkpoint_dir= --ckpt_file=
else
	echo "We don't have that model. Sorry :-("
	exit
fi

echo "The results have been generated! Responses, bleu scores, perplexity scores are stored in $output_file"
echo "Prining the results in the required format:"
python3 do_the_final_printing.py $output_file > final_output.txt
