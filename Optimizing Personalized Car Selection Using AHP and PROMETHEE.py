import numpy as np

def ahp_calculate():
    print('Please enter decision values.')
    price_vs_fuel = float(input('price_vs_fuel: '))
    price_vs_hp = float(input('price_vs_hp: '))
    price_vs_engine_capacity = float(input('price_vs_engine_capacity: '))
    price_vs_max_speed = float(input('price_vs_max_speed: '))
    fuel_vs_hp = float(input('fuel_vs_hp: '))
    fuel_vs_engine_capacity = float(input('fuel_vs_engine_capacity: '))
    fuel_vs_max_speed = float(input('fuel_vs_max_speed: '))
    hp_vs_engine_capacity = float(input('hp_vs_engine_capacity: '))
    hp_vs_max_speed = float(input('hp_vs_max_speed: '))
    engine_capacity_vs_max_speed = float(input('engine_capacity_vs_max_speed: '))

    decisions_matrix = np.array([
        [1, price_vs_fuel, price_vs_hp, price_vs_engine_capacity, price_vs_max_speed],
        [1 / price_vs_fuel, 1, fuel_vs_hp, fuel_vs_engine_capacity, fuel_vs_max_speed],
        [1 / price_vs_hp, 1 / fuel_vs_hp, 1, hp_vs_engine_capacity, hp_vs_max_speed],
        [1 / price_vs_engine_capacity, 1 / fuel_vs_engine_capacity, 1 / hp_vs_engine_capacity, 1, engine_capacity_vs_max_speed],
        [1 / price_vs_max_speed, 1 / fuel_vs_max_speed, 1 / hp_vs_max_speed, 1 / engine_capacity_vs_max_speed, 1]
    ])

    print(decisions_matrix)

    column_totals = np.sum(decisions_matrix, axis=0)
    print("Column Totals:")
    print(column_totals)

    normalized_matrix = decisions_matrix / np.sum(decisions_matrix, axis=0)
    print('Finding the Normalized Matrix')
    print('Normalized_matrix = decisions_matrix / Column Totals')
    print('Normalized_matrix =\n', normalized_matrix)

    criteria_weights = np.mean(normalized_matrix, axis=1)
    criteria_weights_column_matrix = np.reshape(criteria_weights, (len(criteria_weights), 1))
    print('Criteria Weights:')
    print(criteria_weights_column_matrix)

    weighted_matrix = decisions_matrix * criteria_weights
    print('\nWeighted Matrix:')
    print(weighted_matrix)

    row_totals = np.sum(weighted_matrix, axis=1)
    print('\nRow Totals:')
    print(row_totals)

    normalized_row_totals = row_totals / criteria_weights
    print('\nNormalized Row Totals:')
    print(normalized_row_totals)

    lambda_max = np.mean(normalized_row_totals)
    print(lambda_max)
    n = len(decisions_matrix)
    consistency_index = (lambda_max - n) / (n - 1)
    print('Consistency Index:', consistency_index)
    consistency_ratio = consistency_index / 1.115
    print('Consistency Ratio:', consistency_ratio)
    if consistency_ratio <= 0.1:
        print("\nCriteria weights are acceptable.")
    else:
        print("\nCriteria weights are not acceptable.")

    return criteria_weights


def promethee_calculate(criteria_weights):
    with open("cardata.txt", "r") as file:
        lines = file.readlines()

    header = lines.pop(0)

    car_data = []
    for line in lines:
        values = line.strip().split("\t")
        car_info = {
            "brand": values[0],
            "model": values[1],
            "price": float(values[2].replace(".", "").replace(",", ".")),
            "fuel": float(values[3].replace(",", ".")),
            "hp": int(values[4]),
            "engine_capacity": int(values[5]),
            "max_speed": int(values[6])
        }
        car_data.append(car_info)

    selected_body_type = 'SUV'

    selected_body_type_cars = []
    for car in car_data:
        selected_body_type_cars.append({
            "price": car["price"],
            "fuel": car["fuel"],
            "hp": car["hp"],
            "engine_capacity": car["engine_capacity"],
            "max_speed": car["max_speed"]
        })

    print(f"{selected_body_type} vehicles:")
    for car in selected_body_type_cars:
        print(
            f"Price: {car['price']:,} TL, Fuel: {car['fuel']}, HP: {car['hp']}, Engine Capacity: {car['engine_capacity']}, Max Speed: {car['max_speed']}")
    selected_body_type_array = np.array([
        [car["price"], car["fuel"], car["hp"], car["engine_capacity"], car["max_speed"]]
        for car in selected_body_type_cars
    ])
    selected_body_type_cars = []
    car_names = []
    for car in car_data:
        selected_body_type_cars.append([
            car["price"],
            car["fuel"],
            car["hp"],
            car["engine_capacity"],
            car["max_speed"]
        ])
        car_names.append(f"{car['brand']} {car['model']}")
    print(selected_body_type_cars)

    selected_body_type_array = np.array(selected_body_type_cars, dtype=object)

    print(f"{selected_body_type} vehicles:")
    print(selected_body_type_array)
    max_values = np.amax(selected_body_type_array, axis=0)
    min_values = np.amin(selected_body_type_array, axis=0)
    print('Max Values:', max_values)
    print('Min Values:', min_values)

    selected_body_type_array = np.array(selected_body_type_cars, dtype=object)
    for i in range(2):
        col = selected_body_type_array[:, i]
        max_val = np.max(col)
        min_val = np.min(col)
        selected_body_type_array[:, i] = (max_val - col) / (max_val - min_val)

    for i in range(2, selected_body_type_array.shape[1]):
        col = selected_body_type_array[:, i]
        max_val = np.max(col)
        min_val = np.min(col)
        selected_body_type_array[:, i] = (col - min_val) / (max_val - min_val)

    print(f"{selected_body_type} vehicles (normalized):")
    print(selected_body_type_array, '\n\n\n\n')
    selected_body_type_array2 = np.column_stack((selected_body_type_array, np.array(car_names)))
    print(selected_body_type_array2)

    results = []
    operation_info = []

    for i in range(selected_body_type_array.shape[0]):
        for j in range(selected_body_type_array.shape[0]):
            if i != j:
                results.append(selected_body_type_array[i, :] - selected_body_type_array[j, :])
                operation_info.append((i + 1, j + 1))

    threshold = 0.0
    probability_matrix = np.maximum(np.array(results), 0)

    for i, ((result, probability), (row1, row2)) in enumerate(
            zip(zip(results, probability_matrix), operation_info),
            1):
        print(f"{i}. subtraction operation (Row {row1} - Row {row2}):")
        print("Subtraction Result:")
        print(result)
        print("Probability Matrix:")
        print(probability)
        print()

    probability_matrix = np.vstack(
        [probability for (result, probability), _ in zip(zip(results, probability_matrix), operation_info)])

    print("Probability Matrix:")
    print(probability_matrix)
    reference_vector = criteria_weights

    product_matrix = np.array([probability_row * reference_vector for probability_row in probability_matrix])

    print("Product Matrix:")
    print(product_matrix)

    total_matrix = np.sum(product_matrix, axis=1)

    print("Total Matrix:")
    print(total_matrix)
    print(len(total_matrix))
    print(len(selected_body_type_cars))
    n = (len(selected_body_type_cars))

    non_zero_elements = list(total_matrix)

    matrix = []

    count = 0
    for i in range(n):
        row = []
        for j in range(n):
            if i == j:
                row.append(0)
            else:
                row.append(non_zero_elements[count])
                count += 1
        matrix.append(row)

    for row in matrix:
        print(row)
    column_averages = []
    row_averages = []

    for i in range(n):
        non_zero_elements_sum = sum(matrix[i][j] for j in range(n) if j != i)
        non_zero_elements_count = n - 1
        average = non_zero_elements_sum / non_zero_elements_count if non_zero_elements_count != 0 else 0
        print(f"Row {i + 1} average: {average}")
        row_averages.append(average)
    print('\n\n\n\n')
    for j in range(n):
        non_zero_elements_sum = sum(matrix[i][j] for i in range(n) if i != j)
        non_zero_elements_count = n - 1
        average = non_zero_elements_sum / non_zero_elements_count if non_zero_elements_count != 0 else 0
        print(f"Column {j + 1} average: {average}")
        column_averages.append(average)
    print(column_averages)
    print(row_averages)
    column_averages = np.array(column_averages)
    row_averages = np.array(row_averages)
    FINAL_VALUES = list(row_averages - column_averages)
    print(FINAL_VALUES)
    print(len(FINAL_VALUES))
    result_list_sorted = sorted(FINAL_VALUES, key=lambda x: abs(1 - x))

    print(result_list_sorted)

    ranking = {value: order for order, value in enumerate(FINAL_VALUES, start=1)}

    result_determination = list(selected_body_type_array2)
    final_list = []

    for value in result_list_sorted:
        print(f"{value} is at {ranking[value]}. position in the final values.")
        value2 = int(ranking[value])
        final_list.append(value2)
    print(final_list)
    print(result_determination)
    car_names = [car[-1] for car in result_determination]
    final_list2 = []
    print(final_list2)

    for car_name in car_names:
        final_list2.append(car_name)

    print(len(final_list2))
    print(len(final_list))

    for index in final_list:
        print(final_list2[index - 1])

criteria_weights = ahp_calculate()
promethee_calculate(criteria_weights)
