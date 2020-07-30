(
s.options.device_("Scarlett 6i6 USB");
s.waitForBoot({
	var train = {
		arg fm_json, analysis_json, analysis_name, nSteps, shape;
		Task({
			var timestamp = Date.localtime.stamp;
			//var dir = "/Users/ted/Desktop/SCD/flucoma/nn fm/200724_01/%".format(timestamp);
			var dir = "%%_%_%_shape=%".format(PathName(analysis_json).pathOnly,Date.localtime.stamp,nSteps,analysis_name,shape);
			//var fm_json = "/Users/ted/Desktop/SCD/flucoma/nn fm/200723_01/200718_202553_nn_fm_nSteps=30_fm.json";
			//var analysis_json = "/Users/ted/Desktop/SCD/flucoma/nn fm/200723_01/200718_202553_nn_fm_nSteps=30_entire_analysis.json";
			//var analysis_name = "entire_analysis";

			// read
			var fm = FluidDataSet(s,(\fm++UniqueID.next).asSymbol);
			var fm_norm;
			var fm_norm_ds;
			var analysis;
			var analysis_norm;
			var analysis_norm_ds;
			var hidden_act, output_act, activation_ints, maxIter, net;
			var run_fit;

			"------- dir:           %".format(dir).postln;
			"------- fm json:       %".format(fm_json).postln;
			"------- analysis json: %".format(analysis_json).postln;
			"------- name:          %".format(analysis_name).postln;
			"------- nSteps:        %".format(nSteps).postln;
			"------- shape:         %".format(shape).postln;
			"".postln;

			File.mkdir(dir);

			s.sync;
			fm.read(fm_json);
			s.sync;
			//fm_stand = FluidStandardize(s);
			//s.sync;
			//fm_stand_ds = FluidDataSet(s,\fm_stand);
			//s.sync;
			//fm_stand.fitTransform(fm,fm_stand_ds,{"done".postln;});
			//s.sync;
			//fm_stand.write("/Users/ted/Desktop/SCD/flucoma/nn fm/200718_01/%_fm_stand_nPoints=30.json".format(timestamp));
			//s.sync;
			fm_norm = FluidNormalize(s);
			s.sync;
			fm_norm_ds = FluidDataSet(s,(\fm_norm++UniqueID.next).asSymbol);
			s.sync;
			fm_norm.fitTransform(fm,fm_norm_ds/*,{"done".postln;}*/);
			s.sync;
			fm_norm.write("%/%_fm_norm_nSteps=%.json".format(dir,timestamp,nSteps));
			s.sync;

			// analysis data
			analysis = FluidDataSet(s,(\analysis++UniqueID.next).asSymbol);
			s.sync;
			analysis.read(analysis_json);
			s.sync;
			/*analysis_stand = FluidStandardize(s);
			s.sync;
			analysis_stand_ds = FluidDataSet(s,\analysis_stand);
			s.sync;
			analysis_stand.fitTransform(analysis,analysis_stand_ds,{"done".postln;});
			s.sync;
			analysis_stand.write("/Users/ted/Desktop/SCD/flucoma/nn fm/200718_01/%_mfcc_stand_nPoints=30.json".format(timestamp));
			s.sync;*/
			analysis_norm = FluidNormalize(s);
			s.sync;
			analysis_norm_ds = FluidDataSet(s,(\analysis_norm++UniqueID.next).asSymbol);
			s.sync;
			analysis_norm.fitTransform(analysis,analysis_norm_ds/*,{"done".postln;}*/);
			s.sync;
			analysis_norm.write("%/%_%_norm_nSteps=%.json".format(dir,timestamp,analysis_name,nSteps));
			s.sync;

			//fm_norm_ds.print;
			//analysis_norm_ds.print;

			// ============================= hyper params ==========================
			//shape = [40,30,20,10,5];
			//shape = [3,5,3];
			//shape = [10,6];
			//n = FluidMLPRegressor(s);
			hidden_act = "sigmoid";
			output_act = "identity";
			maxIter = 1000;

			// make network ----------------------------------------------------------
			activation_ints = [hidden_act,output_act].collect({
				arg string;
				var return = nil;
				string.switch(
					"sigmoid",{return = FluidMLPRegressor.sigmoid},
					"identity",{return = FluidMLPRegressor.identity},
					"tanh",{return = FluidMLPRegressor.tanh}
				);
				return;
			});

			net = FluidMLPRegressor(s,shape,activation_ints[0],activation_ints[1],0,maxIter,0.0001,batchSize:10);
			s.sync;

			run_fit = {
				arg counter;

				net.fit(analysis_norm_ds,fm_norm_ds,{
					arg error;
					"".postln;
					"------- n steps:  %".format(nSteps).postln;
					"------- analysis: %".format(analysis_name).postln;
					"------- counter:  %".format(counter).postln;
					"------- n iters:  %".format(counter * maxIter).postln;
					"------- shape:    %".format(shape).postln;
					"------- loss:     %".format(error).postln;
					"".postln;

					net.write("%/%_analysis->fm_%_loss=%_nSteps=%_shape=%_hiddenAct=%_outAct=%_nEpochs=%.json".format(
						dir,
						timestamp,
						analysis_name,
						error.round(0.0001).asString.padRight(6,"0"),
						nSteps,
						shape,
						hidden_act,
						output_act,
						counter * maxIter
					),{
						if(error > 0.005,{
							run_fit.(counter+1);
						});
					});
				});
			};

			run_fit.(1);
		}).play;
	};

	/*	~fm_json = "/Users/ted/Desktop/SCD/flucoma/nn fm/200726_01 poisson no median filter/200726_114107_nn_fm_poisson=37542_5k_2500_mfreq<cfreq_indexCalc_fm.json";
	~analysis_json = "/Users/ted/Desktop/SCD/flucoma/nn fm/200726_01 poisson no median filter/200726_114107_nn_fm_poisson=37542_5k_2500_mfreq<cfreq_indexCalc_spec.json";*/

	/*	~large_analysis_json = "/Users/ted/Desktop/SCD/flucoma/nn fm/200726_01 poisson no median filter/200726_114107_nn_fm_poisson=37542_5k_2500_mfreq<cfreq_indexCalc_all_but_chroma.json";*/
	[
		[
			"/Users/ted/Desktop/SCD/flucoma/nn fm/200726_02/200726_114107_nn_fm_poisson=37542_7500_3k_mfreq<cfreq_indexCalc_fm.json",
			"/Users/ted/Desktop/SCD/flucoma/nn fm/200726_02/200726_114107_nn_fm_poisson=37542_7500_3k_mfreq<cfreq_indexCalc_spec.json",
			"spec_filtered_data_7500_3k_indexCalc",5685,[8]
		]/*,
		[
		"/Users/ted/Desktop/SCD/flucoma/nn fm/200726_02/200726_114107_nn_fm_poisson=37542_10k_5k_mfreq<cfreq_indexCalc_fm.json",
		"/Users/ted/Desktop/SCD/flucoma/nn fm/200726_02/200726_114107_nn_fm_poisson=37542_10k_5k_mfreq<cfreq_indexCalc_spec.json",
		"spec_filtered_data_10k_5k_indexCalc",6545,[7]
		],
		[
		"/Users/ted/Desktop/SCD/flucoma/nn fm/200726_02/200726_114107_nn_fm_poisson=37542_10k_5k_mfreq<cfreq_indexCalc_fm.json",
		"/Users/ted/Desktop/SCD/flucoma/nn fm/200726_02/200726_114107_nn_fm_poisson=37542_10k_5k_mfreq<cfreq_indexCalc_spec.json",
		"spec_filtered_data_10k_5k_indexCalc",6545,[6]
		],
		[
		"/Users/ted/Desktop/SCD/flucoma/nn fm/200726_02/200726_114107_nn_fm_poisson=37542_10k_5k_mfreq<cfreq_indexCalc_fm.json",
		"/Users/ted/Desktop/SCD/flucoma/nn fm/200726_02/200726_114107_nn_fm_poisson=37542_10k_5k_mfreq<cfreq_indexCalc_spec.json",
		"spec_filtered_data_10k_5k_indexCalc",6545,[5]
		],
		[
		"/Users/ted/Desktop/SCD/flucoma/nn fm/200726_02/200726_114107_nn_fm_poisson=37542_10k_5k_mfreq<cfreq_indexCalc_fm.json",
		"/Users/ted/Desktop/SCD/flucoma/nn fm/200726_02/200726_114107_nn_fm_poisson=37542_10k_5k_mfreq<cfreq_indexCalc_spec.json",
		"spec_filtered_data_10k_5k_indexCalc",6545,[10,5]
		],
		[
		"/Users/ted/Desktop/SCD/flucoma/nn fm/200726_02/200726_114107_nn_fm_poisson=37542_10k_5k_mfreq<cfreq_indexCalc_fm.json",
		"/Users/ted/Desktop/SCD/flucoma/nn fm/200726_02/200726_114107_nn_fm_poisson=37542_10k_5k_mfreq<cfreq_indexCalc_spec.json",
		"spec_filtered_data_10k_5k_indexCalc",6545,[11,4]
		],
		[
		"/Users/ted/Desktop/SCD/flucoma/nn fm/200726_02/200726_114107_nn_fm_poisson=37542_10k_5k_mfreq<cfreq_indexCalc_fm.json",
		"/Users/ted/Desktop/SCD/flucoma/nn fm/200726_02/200726_114107_nn_fm_poisson=37542_10k_5k_mfreq<cfreq_indexCalc_spec.json",
		"spec_filtered_data_10k_5k_indexCalc",6545,[9,4]
		],
		[
		"/Users/ted/Desktop/SCD/flucoma/nn fm/200726_02/200726_114107_nn_fm_poisson=37542_10k_5k_mfreq<cfreq_indexCalc_fm.json",
		"/Users/ted/Desktop/SCD/flucoma/nn fm/200726_02/200726_114107_nn_fm_poisson=37542_10k_5k_mfreq<cfreq_indexCalc_spec.json",
		"spec_filtered_data_10k_5k_indexCalc",6545,[7,4]
		]*/
	].do({
		arg arr;
		var fm_json = arr[0];
		var analysis_json = arr[1];
		var analysis_name = arr[2];
		var nSteps = arr[3];
		var shape = arr[4];
		train.(fm_json,analysis_json,analysis_name,nSteps,shape);
	});
});
)