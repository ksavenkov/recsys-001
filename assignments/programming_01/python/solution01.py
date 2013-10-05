# make python find our new modules
import sys
sys.path.append("../../../recsys")

import recsys
import dataset
from model import PreferenceModel
from score import cooc_simple, cooc_advanced
from suggest import top_ns
from printer import coursera_pa1_printer

n = 5
given_items = [7443, 602, 280]

def generate_solution(algo, n, given_items, filename):
    # load the dataset
    ds = dataset.DataIO(verbose = True)
    ds.load('../data/ratings.csv')
    given_items = ds.translate_items(given_items)
    
    # build the recommender and generate predictions
    rs = recsys.Recommender(ds, PreferenceModel(verbose = True), algo, top_ns, verbose = True)
    rs.build()
    recs = rs.recommend(given_items, n)

    # save the predictions to file
    output = ds.print_recs(recs, given_items = given_items, printer = coursera_pa1_printer)
    file = open(filename, 'w')
    file.write(output)
    file.close()
    print 'Predictions are written to %s' % filename
   
def main():
    generate_solution(cooc_simple, n, given_items, 'simple.csv') 
    generate_solution(cooc_advanced, n, given_items, 'advanced.csv') 

if __name__ == "__main__":
    main()
