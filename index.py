import geocoder
import csv

f = open('latlng.csv','w')
reader = csv.reader(open("train/train.tsv"), delimiter="	")
flats = []
for xi in reader: flats.append(xi)

flats = flats[2506:]
i = 2506
for flat in flats:
	g = geocoder.google(flat[4])
	print i
	print (g.latlng)
	i += 1
	f.write(str(g.latlng) + "\n")
f.close()
