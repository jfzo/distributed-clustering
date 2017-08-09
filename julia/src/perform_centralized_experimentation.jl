push!(LOAD_PATH, pwd())
using centralized_experiment

DATA_PATH = "./TDT2/tdt2_tfidf_top30.csv";
LABEL_PATH = "./TDT2/tdt2_tfidf_top30.csv.labels";
println("TDT2 top30")
println(snn_grid_evaluation(DATA_PATH, LABEL_PATH))

println("RCV1 top30")
DATA_PATH = "./RCV1/reuters_single_tfidf_top30.csv";
LABEL_PATH = "./RCV1/reuters_single_tfidf_top30.csv.labels";
println(snn_grid_evaluation(DATA_PATH, LABEL_PATH))

println("20NG")
DATA_PATH = "./20newsgroups/20ng_tfidf_cai.csv";
LABEL_PATH = "./20newsgroups/20ng_tfidf_cai.csv.labels";
println(snn_grid_evaluation(DATA_PATH, LABEL_PATH))

println("20NG top10")
DATA_PATH = "./20newsgroups/20ng_tfidf_cai_top10.csv";
LABEL_PATH = "./20newsgroups/20ng_tfidf_cai_top10.csv.labels";
println(snn_grid_evaluation(DATA_PATH, LABEL_PATH))
