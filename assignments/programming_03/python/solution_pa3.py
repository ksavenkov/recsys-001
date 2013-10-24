# make python find our new modules
import sys
sys.path.append("../../../recsys")

from score import user_based_knn, cosine
from dataset import DataIO
from model import UserModel

ratings_file = '../data/ratings.csv'
items_file = '../data/movie-titles.csv'
NN = 30
answer_file = 'part_1.csv'

# part 1

data = DataIO()
data.load(ratings_file, items_file = items_file)
model = UserModel(normalize = True)
model.build(data)

inputs = [(4169,161),
		(4169,36955),
		(4169,453),
		(4169,857),
		(4169,238),
		(5399,1891),
		(5399,14),
		(5399,187),
		(5399,602),
		(5399,629),
		(3613,329),
		(3613,604),
		(3613,134),
		(3613,1637),
		(3613,278),
		(1873,786),
		(1873,2502),
		(1873,550),
		(1873,1894),
		(1873,1422),
		(4914,268),
		(4914,36658),
		(4914,786),
		(4914,161),
		(4914,854)]

file = open(answer_file,'w')
file.write('\n'.join(
            ['%d,%d,%.4f,%s' % (
                u, 
                i, 
                user_based_knn(model, NN, [data.new_user_idx(u)], 
                                          [data.new_item_idx(i)], cosine, promote_users = True, normalize = 'centered')[0], 
                data.title(i))
                    for (u,i) in inputs]))
file.close()
