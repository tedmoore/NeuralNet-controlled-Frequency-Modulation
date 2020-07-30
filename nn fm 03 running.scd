(
s.waitForBoot({
	Task({

		var fm_norm_path = "/Users/ted/Desktop/SCD/flucoma/nn fm/200726_01 poisson no median filter/200726_172723_4584_spec_filtered_data/200726_172723_fm_norm_nSteps=4584.json";
		var analysis_norm_path = "/Users/ted/Desktop/SCD/flucoma/nn fm/200726_01 poisson no median filter/200726_172723_4584_spec_filtered_data/200726_172723_spec_filtered_data_norm_nSteps=4584.json";
		var nn_path = "/Users/ted/Desktop/SCD/flucoma/nn fm/200726_01 poisson no median filter/200726_172723_4584_spec_filtered_data/200726_172723_melbands->fm_spec_filtered_data_nSteps=4584_shape=[ 6 ]_hiddenAct=sigmoid_outAct=identity_nEpochs=31800_loss=0.0499.json";

		var analysis_size = 13;

		var test_buf, net, pitching_bus, catching_bus, input_buf0, input_bu1, output_buf0, fm_mins;
		var fm_maxes, analysis_mins, analysis_maxes, fm_ranges, analysis_ranges;
		var amp_bus, fm_json, analysis_json;

		s.sync;

		net = FluidMLPRegressor();
		pitching_bus = Bus.control(s);
		catching_bus = Bus.control(s);
		input_buf0 = Buffer.alloc(s,1,analysis_size);
		output_buf0 = Buffer.alloc(s,3);

		fm_json = JSONFileReader.read(fm_norm_path);
		analysis_json = JSONFileReader.read(analysis_norm_path);
		amp_bus = Bus.control(s);

		~input_buf1 = Buffer.alloc(s,analysis_size);

		fm_mins = fm_json.at("data_min").asFloat;
		fm_maxes = fm_json.at("data_max").asFloat;
		fm_ranges = fm_maxes - fm_mins;

		analysis_mins = analysis_json.at("data_min").asFloat;
		analysis_maxes = analysis_json.at("data_max").asFloat;
		analysis_ranges = analysis_maxes - analysis_mins;

		//test_buf = Buffer.readChannel(s,"/Volumes/Ted's 10TB My Book (June 2020)/PROJECT FILES/machine learning/Training Data/Audio/a test file.wav",channels:[0]);
//
		test_buf = Buffer.readChannel(s,"/Volumes/Ted's 10TB My Book (June 2020)/PROJECT FILES/machine learning/Training Data/Audio/basson mixed activity for testing (complete).wav",channels:[0]);
		//test_buf = Buffer.readChannel(s,"/Volumes/Ted's 10TB My Book (June 2020)/PROJECT FILES/machine learning/Training Data/Audio/quick brown fox.wav",channels:[0]);
		//test_buf = Buffer.readChannel(s,"/Volumes/Ted's 10TB My Book (June 2020)/SOUND DESIGNS/_EURORACK SOUNDS/200613 eurorack 01/_bounces/200613 eurorack 01 last 10 min excerpt.wav",channels:[0]);

		//test_buf = Buffer.readChannel(s,"/Users/ted/Documents/_CREATING/_PROJECT FILES/wet ink/sounds 2/200606 eurorack 60 hz noise.wav",channels:[0]);

		//test_buf = Buffer.readChannel(s,"/Users/ted/Documents/_CREATING/_PROJECT FILES/wet ink/sounds 2/slowed down eurorack with chromagram data.wav",channels:[0]);

		[fm_mins,fm_maxes,fm_ranges].postln;
		[analysis_mins,analysis_maxes,analysis_ranges].postln;

		s.sync;

		net.read(nn_path);

		s.sync;

		net.inBus_(pitching_bus);
		net.outBus_(catching_bus);
		net.inBuffer_(~input_buf1);
		net.outBuffer_(output_buf0);

		~trig_rate = 25;
		~example_dur = 5;
		//s.record;
		s.sync;

		~inSynth = {
			var sig = PlayBuf.ar(1,test_buf,1,0,rrand(0,test_buf.numFrames),1);
			//var mfcc = FluidMFCC.kr(sig,40)[1..39];
			var spec = FluidSpectralShape.kr(sig);
			var pitch = FluidPitch.kr(sig);
			var loudness = FluidLoudness.kr(sig);
			var zc = A2K.kr(ZeroCrossing.ar(sig));
			var senseDis = SensoryDissonance.kr(FFT(LocalBuf(2048),sig));
			//var melbands = FluidMelBands.kr(sig,maxNumBands:40);
			var trig = Impulse.kr(~trig_rate);
			var flat_trig;
			//var vector = mfcc ++ spec ++ pitch;
			//var vector = spec ++ pitch;
			var vector = spec ++ pitch ++ loudness ++ [zc,senseDis];

			//sig = sig * LFPulse.kr((~example_dur * 2).reciprocal).lag(0.03);

			Out.kr(amp_bus,DelayN.kr(Amplitude.kr(sig),~trig_rate.reciprocal,~trig_rate.reciprocal));



			vector = (vector - analysis_mins) / analysis_ranges;

			vector = Median.kr(31,vector);

			RecordBuf.kr(vector,input_buf0);

			flat_trig = FluidBufFlatten.kr(input_buf0,~input_buf1,trig:trig);

			Out.kr(pitching_bus,Done.kr(flat_trig));
			DelayN.ar(sig,~trig_rate.reciprocal,~trig_rate.reciprocal);
			//sig;
		}.play(net.synth,addAction:\addBefore);

		~outSynth = {
			var max_del = 8;
			var inTrig = In.kr(catching_bus);
			var outs = 3.collect({
				arg i;
				Index.kr(output_buf0,i);
			});
			var sig;
			var cfreq, mfreq, index;
			var del_time = LFDNoise3.kr(2).range(0,1).pow(2) * max_del;

			outs = Median.kr(31,outs);

			outs = (outs * fm_ranges) + fm_mins;

			outs = outs.lag(~trig_rate.reciprocal);

			outs = outs ++ [In.kr(amp_bus)];

			outs.poll;

			cfreq = outs[0].clip(20,20000);
			mfreq = outs[1].clip(20,20000);
			index = max(outs[2],0);

			sig = SinOsc.ar(cfreq + SinOsc.ar(mfreq,0,mfreq * index));

			sig = sig * outs[3];
			Out.ar(1,sig);
		}.play;
	}).play;
});
)

//s.stopRecording;