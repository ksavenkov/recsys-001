# make python find our new modules
import sys
sys.path.append("../../recsys")

from score import user_based_knn, pearson
from dataset import DataIO
from model import UserModel
from suggest import top_ns

ratings_file = 'ratings.csv'
given_users = [3867,860]
NN = 5
n = 3
part_1_file = 'part_1.csv'
part_2_file = 'part_2.csv'

# part 1

data = DataIO()
data.load(ratings_file)
model = UserModel(normalize = False)
model.build(data)

given_users = data.translate_users(given_users)
given_items = range(data.num_items())

R = user_based_knn(model, NN, given_users, given_items, pearson, promote_users = False)
recs = top_ns(R,n,keep_order = True)

file = open(part_1_file,'w')
file.write('\n'.join(['%d %.3f' % (data.old_item_idx(i),s) for u in recs for (i,s) in u]))
file.close()

# part 2

R = user_based_knn(model, NN, given_users, given_items, pearson, promote_users = False, exclude_seen = False, normalize = True)
recs = top_ns(R,n,keep_order = True)

file = open(part_2_file,'w')
file.write('\n'.join(['%d %.3f' % (data.old_item_idx(i),s) for u in recs for (i,s) in u]))
file.close()


