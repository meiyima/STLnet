f = open('airquality.csv') #Air Quality Data Path
from datetime import datetime
import torch
stations = {}
top = f.readline()
for row in f.readlines():
    rowsp = row.split(',')
    try:
        station_id = int(rowsp[0])
        timeh = int((datetime.strptime(rowsp[1], '%Y-%m-%d %H:%M:%S') - datetime(2014, 5, 1)).total_seconds() / 3600)
        pm25 = float(rowsp[2])
        if station_id in stations:
            stations[station_id][timeh] = pm25
        else:
            stations[station_id] = torch.zeros(8760)
            stations[station_id][timeh] = pm25
    except:
        pass

bjst = []
for stationid in stations:
    city_idx = stationid // 1000
    if city_idx==1:
        bjst.append(stations[stationid])
bjs = torch.stack(bjst, dim=0)
torch.save(bjs, 'beijing_pm25.dat')