RR = R --vanilla --slave --quiet


../Processed/test_scrubbed.csv ../Processed/train_scrubbed.csv :  \
    ../Raw/test.csv ../Raw/train.csv scrub.r
	$(RR) < scrub.r

