import csv
import numpy

def write_top_n_titles (titles_list, metric_array, n, filename):
    #sort titles by metric, retrieve indexes for top-n
    top_n_metric_indices = metric_array.argsort()[::-1][:n]
    #fetch title ids
    top_n_metric_titles = [titles_list[i][0] for i in top_n_metric_indices]
    #write them to file
    file = open(filename, 'w')
    for t in top_n_metric_titles:
        file.write('%s\n' % t)
    file.close()
   
#for top-n
n = 5
raw_data = []

#read ratings to a list of rows
with open('matrix.csv','rbU') as csvfile:
    movie_reader = csv.reader(csvfile, delimiter=',')
    for row in movie_reader:
        raw_data.append(row)

num_titles = len(raw_data[0]) - 1 
num_users = len(raw_data) - 1

print 'Read ratings for %d movies from %d users' % (num_titles, num_users)

#create lists of titles and users, skip the first raw and first column
titles = [t.split(':') for t in raw_data[0][1:]]
users = [r[0] for r in raw_data[1:]]

ratings = numpy.array([[int(c) if c!='' else numpy.nan for c in r[1:]] for r in raw_data[1:]])

#1. Mean ratings
masked_nans = numpy.ma.masked_array(ratings,numpy.isnan(ratings))
mean_ratings = numpy.mean(masked_nans,0)
#write top-n to a file
write_top_n_titles(titles, mean_ratings, n, 'top_%d_mean.txt' % n)

#2. % of ratings 4+
all_ratings = numpy.ma.count(masked_nans,0)
masked_low = numpy.logical_or(numpy.ma.masked_array(ratings, ratings < 4),masked_nans)
high_ratings = numpy.ma.count(masked_low, 0)
best_ratio = numpy.true_divide(high_ratings,all_ratings)
#write top-n to a file
write_top_n_titles(titles, best_ratio, n, 'top_%d_best.txt' % n)

#3 rating count
#already computed, just write in file
write_top_n_titles(titles, all_ratings, n, 'top_%d_count.txt' % n)

#4 cooccurence with Start Wars [0]
star_wars_users_mask = masked_nans[:,0].mask # False = rated 0th movie
star_wars_users_mask_matrix = numpy.array([star_wars_users_mask,]*num_titles).T

cooc_star_wars_mask = numpy.logical_or(masked_nans.mask, star_wars_users_mask_matrix)
cooc_star_wars_count = numpy.ma.count(numpy.ma.masked_array(ratings,cooc_star_wars_mask),0) 
cooc_star_wars_ratio = cooc_star_wars_count / float(numpy.ma.count(masked_nans[:,0]))
#write top-n to a file
n = 6
write_top_n_titles(titles, cooc_star_wars_ratio, n, 'top_%d_star_wars.txt' % n)

