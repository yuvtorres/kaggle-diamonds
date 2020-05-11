#Programa para predecir el valor de los diamantes
import argparse
import sys
sys.path.insert(0, 'src/')
from src import data_cleaning as d_c
from src import run_models as r_m


def main():
    # Define the description of the programa
    file1 = open("src/description.txt","r")
    description_text = '\n'.join( file1.readlines() )
    file1.close()
    parser = argparse.ArgumentParser(description=description_text)

    # optional arguments to execute the cleaning of the data
    parser.add_argument('--data_c_t', action='store_true',
                help=''' (default: False) Make the transformation of the data:
                take out outlayers, dummies geneation, numeric encoding, PCA,
                and normalize. And save them in output.\n''')

    # optional arguments for the analysis
    parser.add_argument('--pre_graph', metavar='G', type=int,nargs='+', 
			help='Create analysis graph:\n 1->Corr\n2 -> Matrix\n3 -> Both ')
    parser.add_argument('--models', metavar='Y', type=int,nargs='+',
			help='''execute a particular model creating the predictions and a
            graph of the performance:\n 1->Linear\n2 -> SRV_rbf\n3 ->
            RandomForest_reg\n4 -> GradientBoostingRegressor\n5->SGDRegressor\n
            6->HistGradientBoostingRegressor''')
    parser.add_argument('--resume', action='store_true', 
            help='If active a resume of the models is generated \n')

    parser.add_argument('--hist_deep', metavar='hi',type=int, nargs='+',default=1,\
            help='''Execute an iterative process over the Histogram-based
            Gradient Boosting regression search for a best solution ''')

    args = parser.parse_args()

	# Data cleaning and transformation
    if args.data_c_t:
	    d_c.cleaning_data()

    # create some previuos graphs
    if args.pre_graph:
        d_c.pre_graph(args.pre_graph[0])

    # Run models
    if args.models:
        r_m.run_model(args.models[0])

    if args.resume:
        r_m.resume()

    if args.hist_deep:
        if args.hist_deep[0]==1:
            r_m.h_itera()
        else:
            for k in range(args.hist_deep[0]):
                r_m.h_itera(str(k))


if __name__=="__main__": main()
