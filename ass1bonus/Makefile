FILES = Assignment1Bonus.m LoadBatch.m ComputeCost.m EvaluateClassifier.m MiniBatchGD.m  ComputeCostSVM.m EvaluateClassifierSVM.m MiniBatchGDSVM.m



all: merged_code.m


merged_code.m : $(FILES)
	@for filename in $(FILES); do					\
	    echo -e "\n\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%";	\
	    echo  "%%Content of file : $$filename";			\
	    echo  "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n\n";	\
	    cat "$$filename";						\
	done > merged_code.m ;
#cat  $(FILES) > merged_code.m
