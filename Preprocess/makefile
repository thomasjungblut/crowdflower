RR = R --vanilla --slave --quiet

all : ../Processed/test_scrubbed.csv ../Processed/train_scrubbed.csv

../Processed/test_scrubbed.csv : ../Raw/test.csv scrub.r
	$(RR) --args $< $@ < scrub.r

../Processed/train_scrubbed.csv : ../Raw/train.csv scrub.r
	$(RR) --args $< $@ < scrub.r

