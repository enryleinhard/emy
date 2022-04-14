import pymongo
import pandas as pd
import numpy as np
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import copy
import requests
import json

SECURITY_STRING = "mongodb+srv://anagata:adminanagata@cluster0.penfp.mongodb.net/myFirstDatabase?retryWrites=true&w=majority"
client = pymongo.MongoClient(SECURITY_STRING)
DataBase = client['proddb']


def optimize_expected(date_dropoff):

    order_df = pd.DataFrame(DataBase.orders.find({'date_dropoff': date_dropoff}))
    dist_df = pd.DataFrame(DataBase.distance.find())
    order_df['date_dropoff'] = order_df['date_dropoff'].astype(str)
    date_dropoff = order_df['date_dropoff'][0]

    order_df = order_df.groupby(['orig_store', 'dest_store'])['total_weight'].sum()
    order_df = order_df.to_frame().reset_index()
    res_df = order_df.merge(dist_df, on=["orig_store", "dest_store"])
    res_df = res_df.sort_values('total_weight', ascending=False).reset_index(drop=True)

    store_names = res_df['dest_store'].to_list()
    store_names.insert(0, 'Warehouse')

    #convert list to dict
    def convert(lst):
        res_dct = {lst[i]: lst[i + 1] for i in range(0, len(lst), 2)}
        return res_dct

 #generate dist matrix dict
    dist_adj_list = {}
    dist_mylist = []

    def add_store(node):
        if node not in dist_mylist:
            dist_mylist.append(node)
        else:
            pass

    def add_dist(node1, node2, weight):
        temp = []
        if node1 in dist_mylist and node2 in dist_mylist:
            if node1 not in dist_adj_list:
                temp.append([node2, weight])
                dist_adj_list[node1] = temp

            elif node1 in dist_adj_list:
                temp.extend(dist_adj_list[node1])
                temp.append([node2, weight])
                dist_adj_list[node1] = temp

        else:
            pass

        for node in dist_adj_list:
            [convert(i) for i in dist_adj_list[node]]


    # Adding stores
    for i in store_names:
        add_store(i)

    # Adding distances
    for i in range(len(dist_df)):
        add_dist(dist_df.loc[i]['orig_store'], dist_df.loc[i]
                ['dest_store'], dist_df.loc[i]['dist_km'])

    dist_dict = {}

    for node in dist_adj_list:

        dist_dict[f'{node}'] = []
        for i in dist_adj_list[node]:
            i = convert(i)
            for j, k in i.items():
                d = {
                    f'{j}': k
                }
                dist_dict[f'{node}'].append(d)

    # generate weight matrix dict
    weight_adj_list = {}
    weight_mylist = []

    def add_store(node):
        if node not in weight_mylist:
            weight_mylist.append(node)
        else:
            pass

    def add_weight(node1, node2, weight):
        temp = []
        if node1 in weight_mylist and node2 in weight_mylist:
            if node1 not in weight_adj_list:
                temp.append([node2, weight])
                weight_adj_list[node1] = temp

            elif node1 in weight_adj_list:
                temp.extend(weight_adj_list[node1])
                temp.append([node2, weight])
                weight_adj_list[node1] = temp

        else:
            pass

        for node in weight_adj_list:
            [convert(i) for i in weight_adj_list[node]]

    # Adding stores
    for i in store_names:
        add_store(i)

    # Adding weights
    for i in range(len(res_df)):
        add_weight(res_df.loc[i]['orig_store'], res_df.loc[i]
                ['dest_store'], res_df.loc[i]['total_weight'])

    weight_dict = {}

    for node in weight_adj_list:
        # print(node)
        weight_dict[f'{node}'] = []
        for i in weight_adj_list[node]:
            i = convert(i)
            for j, k in i.items():
                d = {
                    f'{j}': k
                }
                weight_dict[f'{node}'].append(d)


    #adj matrix generator
    def weighted_adjmatrix(adjlist, nodes):

        matrix = []
        for node in nodes:
            weights = {endnode: float(weight)
                       for w in adjlist.get(node, {})
                       for endnode, weight in w.items()}
            matrix.append([weights.get(endnode, 0) for endnode in nodes])
        matrix = np.array(matrix)

        return matrix + matrix.transpose()


    #dict to df for adj matrix
    weight_adj_matrix = weighted_adjmatrix(weight_dict, nodes=store_names)
    weight_adj_matrix_df = pd.DataFrame(weight_adj_matrix,  columns=store_names, index=store_names)
    
    # weight_adj_matrix_df.to_excel('tes_main_weight_matrix.xlsx', index=False)

    dist_adj_matrix = weighted_adjmatrix(dist_dict, nodes=store_names)
    dist_adj_matrix_df = pd.DataFrame(dist_adj_matrix,  columns=store_names, index=store_names)

    # dist_adj_matrix_df.to_excel('tes_main_dist_matrix.xlsx', index=False)


    # find all possible routes
    def create_data_model(nvehicles):

        data = {}
        data['distance_matrix'] = dist_adj_matrix
        data['num_vehicles'] = nvehicles
        data['depot'] = 0
        return data

    def print_solution(data, manager, routing, solution):

        route_list = []

        for vehicle_id in range(data['num_vehicles']):
            each_route = []
            index = routing.Start(vehicle_id)
            plan_output = 'Route for vehicle {}:\n'.format(vehicle_id+1)
            route_distance = 0

            while not routing.IsEnd(index):

                plan_output += ' {} -> '.format(manager.IndexToNode(index))

                each_route.append(manager.IndexToNode(index))

                previous_index = index
                index = solution.Value(routing.NextVar(index))

                route_distance += routing.GetArcCostForVehicle(
                    previous_index, index, vehicle_id)

            plan_output += '{}\n'.format(manager.IndexToNode(index))
            plan_output += 'Distance of the route: {}km\n'.format(
                route_distance)
            print(plan_output)

            route_list.append(each_route)

        return route_list

    def main(nvehicles):

        data = create_data_model(nvehicles)

        manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                               data['num_vehicles'], data['depot'])

        routing = pywrapcp.RoutingModel(manager)

        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return data['distance_matrix'][from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)

        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        dimension_name = 'Distance'
        routing.AddDimension(
            transit_callback_index,
            0,  # no slack
            3000,  # vehicle maximum travel distance
            True,  # start cumul to zero
            dimension_name)
        distance_dimension = routing.GetDimensionOrDie(dimension_name)
        distance_dimension.SetGlobalSpanCostCoefficient(100)

        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

        solution = routing.SolveWithParameters(search_parameters)

        # print solution
        if solution:
            return print_solution(data, manager, routing, solution)
        else:
            print('No solution found !')

    res = {}

    for i in range(1, 10):
        print()
        print()
        print()
        print('=========================== LOOP',
              i, '===============================')
        print()
        print()
        res[f"{i}"] = main(i)
    

    # find number of vehicles needed, routes for each vehicle, and total co2 emission for the most optimized route
    def expected_calc_co2(result_dict, weight_matrix, dist_matrix):

        froad = 0.096
        total_co2_list = []
        output_dict = {"date": date_dropoff, "n_vehicles": '',
                       "routes": '', "total_expected_co2": ''}

        for n in result_dict:
            co2_per_route = 0

            for n_route in result_dict[n]:
                temp_weight = 0

                for idx in range(1, len(n_route)):
                    temp_weight += weight_matrix.iat[(n_route[0]),
                                                     (n_route[idx])]

                for idx in range(1, len(n_route)):
                    updated_weight = temp_weight - \
                        (weight_matrix.iat[(n_route[0]), (n_route[idx])])
                    each_co2 = (temp_weight.astype(float)/1000) * (dist_matrix.iat[(n_route[idx-1]),(n_route[idx])]) * froad
                    co2_per_route += each_co2
                    temp_weight = updated_weight

            total_co2_list.append(co2_per_route)

        # find route with smallest co2 emission
        min_co2_value = min(total_co2_list)
        routes_index = total_co2_list.index(min_co2_value)+1

        output_dict['n_vehicles'] = len(result_dict[f'{routes_index}'])
        output_dict['routes'] = result_dict[f'{routes_index}']
        output_dict['total_expected_co2'] = min_co2_value

        # remove vehicle with no shops in the route
        nonzero_list = []
        for index in range(len(output_dict['routes'])):
            if len(output_dict['routes'][index]) != 1:
                nonzero_list.append(output_dict['routes'][index])

        output_dict['routes'] = nonzero_list
        output_dict['n_vehicles'] = len(nonzero_list)

        return output_dict

    output_dict = expected_calc_co2(res, weight_adj_matrix_df, dist_adj_matrix_df)
    # print(output_dict)

    
    latlong_output_dict = copy.deepcopy(output_dict)

    for index in range(len(latlong_output_dict['routes'])):
        shops_list = []
        for index_shop in latlong_output_dict['routes'][index]:
            shops_list.append(store_names[index_shop])
        latlong_output_dict['routes'][index] = shops_list
    
    latlong_df = pd.DataFrame(DataBase.locations.find())
    
    latlong_route_list = []
    for x in latlong_output_dict['routes']:
        # print(x)
        each_latlong_route_list = []
        for y in x:
            each_shop_dict = {}
            each_shop_dict['shop_name'] = y
            each_shop_dict['lat'] = latlong_df.loc[latlong_df['store_name'] == y, 'latitude'].iloc[0]
            each_shop_dict['long'] = latlong_df.loc[latlong_df['store_name'] == y, 'longitude'].iloc[0]
            # print(each_shop_dict)
            each_latlong_route_list.append(each_shop_dict)
        # print(each_latlong_route_list)
        # print('--------------')
        latlong_route_list.append(each_latlong_route_list)

    latlong_output_dict['routes'] = latlong_route_list

    return latlong_output_dict, output_dict

final_output_dict = optimize_expected("4/5/2022")



def optimize_real(date_dropoff):


    order_df = pd.DataFrame(DataBase.orders.find({'date_dropoff': date_dropoff}))
    dist_df = pd.DataFrame(DataBase.distance.find())
    order_df['date_dropoff'] = order_df['date_dropoff'].astype(str)
    date_dropoff = order_df['date_dropoff'][0]

    order_df = order_df.groupby(['orig_store', 'dest_store'])['total_weight'].sum()
    order_df = order_df.to_frame().reset_index()
    res_df = order_df.merge(dist_df, on=["orig_store", "dest_store"])
    res_df = res_df.sort_values('total_weight', ascending=False).reset_index(drop=True)

    store_names = res_df['dest_store'].to_list()
    store_names.insert(0, 'Warehouse')

    #convert list to dict
    def convert(lst):
        res_dct = {lst[i]: lst[i + 1] for i in range(0, len(lst), 2)}
        return res_dct

    # generate weight matrix dict
    weight_adj_list = {}
    weight_mylist = []

    def add_store(node):
        if node not in weight_mylist:
            weight_mylist.append(node)
        else:
            pass

    def add_weight(node1, node2, weight):
        temp = []
        if node1 in weight_mylist and node2 in weight_mylist:
            if node1 not in weight_adj_list:
                temp.append([node2, weight])
                weight_adj_list[node1] = temp

            elif node1 in weight_adj_list:
                temp.extend(weight_adj_list[node1])
                temp.append([node2, weight])
                weight_adj_list[node1] = temp

        else:
            pass

        for node in weight_adj_list:
            [convert(i) for i in weight_adj_list[node]]

    # Adding stores
    for i in store_names:
        add_store(i)

    # Adding weights
    for i in range(len(res_df)):
        add_weight(res_df.loc[i]['orig_store'], res_df.loc[i]
                ['dest_store'], res_df.loc[i]['total_weight'])

    weight_dict = {}

    for node in weight_adj_list:
        # print(node)
        weight_dict[f'{node}'] = []
        for i in weight_adj_list[node]:
            i = convert(i)
            for j, k in i.items():
                d = {
                    f'{j}': k
                }
                weight_dict[f'{node}'].append(d)


    #adj matrix generator
    def weighted_adjmatrix(adjlist, nodes):

        matrix = []
        for node in nodes:
            weights = {endnode: float(weight)
                        for w in adjlist.get(node, {})
                        for endnode, weight in w.items()}
            matrix.append([weights.get(endnode, 0) for endnode in nodes])
        matrix = np.array(matrix)

        return matrix + matrix.transpose()


    #dict to df for adj matrix
    weight_adj_matrix = weighted_adjmatrix(weight_dict, nodes=store_names)
    weight_adj_matrix_df = pd.DataFrame(weight_adj_matrix,  columns=store_names, index=store_names)



    shopindex_output_dict = copy.deepcopy(final_output_dict[1])
    real_output_dict = copy.deepcopy(final_output_dict[0])

    
    url_real_data = "https://backend-emy.vercel.app/tracker/route"

    payload = json.dumps({
    "date": "4/5/2022"
    })
    headers = {
    'Content-Type': 'application/json'
    }
    response = requests.request("POST", url_real_data, headers=headers, data=payload)
    real_dist = json.loads(response.text)

    real_output_dict['real_distance_traveled'] = real_dist['data']


    def real_calc_co2(weight_matrix, expected_dict, real_dict):
        froad = 0.096
        real_total_co2 = 0
        count = 0
        co2_routes = []

        for n_route in expected_dict['routes']:
            temp_weight = 0
            co2_per_route = 0 

            for idx in range(1, len(n_route)):
                temp_weight += weight_matrix.iat[(n_route[0]), (n_route[idx])]

            for idx in range(1, len(n_route)):
                updated_weight = temp_weight - (weight_matrix.iat[(n_route[0]), (n_route[idx])])
                each_co2 = (temp_weight.astype(float)/1000) * real_dict['real_distance_traveled'][count][idx-1]['dist_km'] * froad

                co2_per_route += each_co2

                temp_weight = updated_weight

            co2_routes.append(co2_per_route)
            count += 1
            # print(co2_per_route)
            real_total_co2 += co2_per_route

        return real_total_co2, co2_routes


    final_dict = {}
    final_dict['date'] = real_output_dict['date']
    final_dict['n_vehicles'] = real_output_dict['n_vehicles']
    final_dict['routes'] = real_output_dict['routes']
    final_dict['total_expected_co2'] = real_output_dict['total_expected_co2']

    final_dict['real_distance_traveled'] = []

    for x in real_output_dict['real_distance_traveled']:
        temp = {'routes': x}
        final_dict['real_distance_traveled'].append(temp)

    co2_real = real_calc_co2(weight_adj_matrix_df, shopindex_output_dict, real_output_dict)

    weights = []
    for n_route in shopindex_output_dict['routes']:
        weight_each_route = 0
        # print(n_route)
        for idx in range(1, len(n_route)):
            weight_each_shop = weight_adj_matrix_df.iat[(n_route[0]), (n_route[idx])]
            weight_each_route += weight_each_shop
        
        weights.append(weight_each_route)

    idx = 0
    for x in final_dict['real_distance_traveled']:
        x['emission'] = co2_real[1][idx]
        x['weight'] = weights[idx]
        idx += 1
    
    final_dict['total_co2_emission'] = co2_real[0]
    
    return final_dict

real_output_dict = optimize_real("4/5/2022")
print("Pushing to Database")

url_update_route = "https://backend-emy.vercel.app/route-optimizer/update-route"

payload = json.dumps({
    "date": "4/5/2022",
    "real_distance_traveled": real_output_dict['real_distance_traveled'],
    "total_CO2_emission": real_output_dict['total_co2_emission']
})
headers = {
  'Content-Type': 'application/json'
}

response = requests.request("POST", url_update_route, headers=headers, data=payload)
response.text