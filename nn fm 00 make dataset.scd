(
~dir = "/Users/ted/Desktop/SCD/flucoma/nn fm/200726_01 poisson no median filter";
File.mkdir(~dir);
~stamp = Date.localtime.stamp;
[30].do({
	arg nSteps;
	//~audio_path = nil;
	SynthMIRNRT(
		[
			[
				\cfreq,
				ControlSpec(20,20000,\exp)
			],[
				\mfreq,
				ControlSpec(20,20000,\exp)
			],[
				\index,
				ControlSpec(0,20,\lin)
			]
		],"%/%_nn_fm_poisson=37542.csv".format(~dir,~stamp),
		SynthDef(\fm_test_nrt,{
			arg cfreq = 20, mfreq = 20, index = 0, outBus = 0;

			// synth stuff
			var sig = SinOsc.ar(cfreq + SinOsc.ar(mfreq,0,index * mfreq));
			//[cfreq,mfreq,index].poll;

			Out.ar(outBus,sig);
		}).writeDefFile,
		"/Volumes/Ted's 10TB My Book (June 2020)/PROJECT FILES/machine learning/Sampling/Poisson Sampling/poisson_sampling_n_dims/generated_samples/poisson_sample_set_ndims=3_npoints=37542_r=0.03_k=20_2020-07-16_20-39-56.csv",
		0.5,0.1,
		nil,//~audio_path,
		{"============== DONE =============".postln;},
		true // verbosity
	);
});
)