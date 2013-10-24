'''Converts input data for the writing assignment 4 to a conventional form 
   used in all programming assignments
'''

import csv 

filename = 'recsys-data-sample-rating-matrix.csv'
ratings_file = 'ratings.csv'
movies_file = 'movies.csv'

def write_table(filename, rows):
    # write movies  
    file = open(filename,'w')
    csv_writer = csv.writer(file, delimiter=',')
    for r in rows:
        csv_writer.writerow(r)
    file.close()

ratings = []
movies = []
with open(filename, 'rbU') as csv_source:
    csv_reader = csv.reader(csv_source, delimiter=',')
    # read user ids, skip the 0th element ("")
    user_ids = [int(u) for u in csv_reader.next()[1:]]
    # read rating lines
    for r in csv_reader:
        # some dances to keep ':'s in the movie title
        movie_bits = r[0].split(':')
        movie_id, movie_title = int(movie_bits[0]), ':'.join(movie_bits[1:]).strip()
        movies.append((movie_id, movie_title))
        ratings.extend([(int(user_id), movie_id, float(rating)) for (user_id,rating) in zip(user_ids,r[1:]) if rating])

    write_table(ratings_file, ratings)
    write_table(movies_file, movies)

