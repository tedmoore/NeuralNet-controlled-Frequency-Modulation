(
~csv_data = CSVFileReader.readInterpret("/Users/ted/Desktop/SCD/flucoma/nn fm/200726_01 poisson no median filter/200726_114107_nn_fm_poisson=37542.csv",startRow:1);
~csv_data.size.postln;
~csv_data = ~csv_data.select({
	arg row;
	//cfreq < 7500      mfreq  < 3000      mfreq  < cfreq       ((index  + 1) * mfreq) < cfreq
	(row[0] < 7500) && (row[1] < 3000) && (row[1] < row[0]) && (((row[2] + 1) * row[1]) < row[0])
});

ArrayToCSV(~csv_data,"/Users/ted/Desktop/SCD/flucoma/nn fm/200726_02/200726_114107_nn_fm_poisson=37542_7500_3k_mfreq<cfreq_indexCalc.csv");
"done";
~csv_data.size.postln;

)

// peek to get a sense of it
(
~keep_indices = [0,1,2] ++ (43..55);
~csv_data = ~csv_data.collect({
	arg row;
	row.atAll(~keep_indices);
});
~normed_data = MinMaxScaler.fit_transform(~csv_data);
~headers = ["cfreq","mfreq","index","spec_centroid",
	"spec_spread",
	"spec_skewness",
	"spec_kurtosis",
	"spec_rolloff",
	"spec_flatness",
	"spec_crest",
	"pitch",
	"pitch_confidence",
	"loudness",
	"loudness_truepeak",
	"zero_crossing",
	"sensory_dissonance"
];
PlotXYColor(~normed_data,{
	arg idx;
	idx.postln;
	~csv_data[idx][0..2].postln;
	~csv_data[idx][3..].postln;
},~headers,slewTime:0);
)