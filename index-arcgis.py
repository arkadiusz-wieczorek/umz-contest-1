import csv
import geocoder

f = open('latlng-arcgis.csv','w')
reader = csv.reader(open("train/train.tsv"), delimiter="	")
flats = []

for xi in reader: 
	flats.append(xi)

i = 1

for flat in flats:
	g = geocoder.arcgis(flat[4])
	print(i)
	if('lat' in g.json and 'lng' in g.json):	
		f.write(str(g.json['lat']) + "," + str(g.json['lng']) + "\n")
	else:
		g2 = geocoder.google(flat[4])
		if(hasattr(g2, 'latlng')): f.write(str(g.latlng) + "\n")
		else: f.write("brak, brak")
	i += 1
f.close()

