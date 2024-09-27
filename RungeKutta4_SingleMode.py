# ----- Import libraries -----
import numpy as np
import scipy.integrate as integrate
import math

# ----- Time constants -----
ending_time = 0.01  # [s]
number_of_steps = 5000  # [-]

# ----- Physical constants -----
# Plate strip material
density_steel = 7850  # [kg/m^3]
poisson_ratio = 0.3  # [-]
elasticity_modulus = 210 * 10 ** 9  # [N/m^2]

# Plate strip geometry
plate_length = 0.5  # [m]
plate_thickness = 0.021  # [m]
plate_area = plate_thickness  # [m^2]
moment_of_inertia = ((plate_length * plate_thickness ** 3) / 12
                     / (1 - poisson_ratio ** 2))  # [m^4]

# Wave
density_water = 1025  # [kg/m^3]
wave_length = 50  # [m]
wave_number = 2 * math.pi / wave_length  # [1/m]
wave_amplitude = 2  # [m]
impact_velocity = 1  # [m/s]

beta = 9.4601  # [1/m]


# ----- Shape functions -----
# Three different shape functions, since the integrands are different
# Regular shape function raised to variable power
def shape_function_one(x, power):
    return ((-1.0178 * math.cos(beta * x) + math.sin(beta * x)
            + 1.0178 * math.cosh(beta * x) - math.sinh(beta * x))) ** power


# Shape function multiplied by square root term
def shape_function_wet_top(x, wetted_length):
    return (((-1.0178 * math.cos(beta * x) + math.sin(beta * x)
            + 1.0178 * math.cosh(beta * x) - math.sinh(beta * x)))
            * (wetted_length ** 2 - x ** 2) ** 0.5)


# Shape function divided by square root term
def shape_function_wet_bottom(x, wetted_length):
    return (((-1.0178 * math.cos(beta * x) + math.sin(beta * x)
            + 1.0178 * math.cosh(beta * x) - math.sinh(beta * x)))
            * (wetted_length ** 2 - x ** 2) ** -0.5)


# ----- Mass equation terms -----
# Structural mass [kg]
def structural_mass():
    power = 2
    value = (density_steel * plate_area *
             integrate.quad(shape_function_one, 0, plate_length, args=(power))[0])

    return value


# Structural stiffness [N/m^2]
def structural_stiffness():
    power = 2
    value = (elasticity_modulus * moment_of_inertia * beta ** 4 *
             integrate.quad(shape_function_one, 0, plate_length, args=(power))[0])

    return value


# Added mass [kg]
def added_mass(wetted_length):
    if wetted_length == 0:
        return 0
    else:
        integral_one = integrate.quad(shape_function_wet_top, 0, wetted_length, args=(wetted_length))[0]
        integral_two = integrate.quad(shape_function_one, 0, wetted_length, args=(1))[0]

        value = density_water * integral_one * integral_two / wetted_length

        return value


# Damping [???]
def damping(wetted_length, wetted_length_change):
    if wetted_length == 0:
        return 0
    else:
        integral_one = integrate.quad(shape_function_wet_bottom, 0, wetted_length, args=(wetted_length))[0]
        integral_two = integrate.quad(shape_function_one, 0, wetted_length, args=(1))[0]

        value = (density_water * wetted_length * wetted_length_change * integral_one
                 * integral_two / wetted_length)

        return value


# Force [N]
def force(wetted_length, wetted_length_change):
    if wetted_length == 0:
        return 0

    else:
        integral_one = integrate.quad(shape_function_wet_bottom, 0, wetted_length, args=(wetted_length))[0]

        value = (density_water * impact_velocity * wetted_length
                 * wetted_length_change * integral_one)

        return value


# ----- Wetted length (Wagner formulation) -----
def wetted_length_wagner(time):
    return 2 * (impact_velocity * time /
                (wave_number ** 2 * wave_amplitude)) ** 0.5


# ----- Mass equation solving  -----
# Matrix/vector generator
# Generates correctly shaped matrix/vector in mass equation term for solving
def matrix_vector_generator(wetted_length, wetted_length_change):
    m_mm = structural_mass()
    k_mm = structural_stiffness()

    a_mm = added_mass(wetted_length)
    d_mm = damping(wetted_length, wetted_length_change)
    f_m = force(wetted_length, wetted_length_change)

    inverse_mass_term = (m_mm + a_mm) ** -1

    matrix = np.array([[0, 1],
                       [-inverse_mass_term * k_mm, -inverse_mass_term * d_mm]])

    vector = np.array([[0], [inverse_mass_term * f_m]])

    return matrix, vector


# Matrix vector multiplier
# Uses dot product to multiply and solve mass equation
def matrix_vector_multiplication(matrix, vector, input_vector):
    result = matrix.dot(input_vector) + vector

    return result


# ----- Overall temporal term a(t) calculator -----
def mode_one_temporal_term(write_file=False):
    # Create even spacing of time points using ending time and no of steps
    time_points = np.linspace(0, ending_time, number_of_steps)

    # Create current guess, starts at [0, 0]
    current_guess = np.array([[0],
                              [0]])

    # Create an empty list to save future guesses
    guesses = []

    # Save a previous wetted length to calculate rate of change
    previous_wetted_length = 0

    # Time step is equal to the first non-zero entry in the time points array
    time_step = time_points[1]

    # Start looping across every time point except the last
    for n in range(len(time_points) - 1):
        # Establish time points: 1 for t_i+1, 0 for t_i, 0.5 for halfway between
        n1_time_point = time_points[n + 1]
        n0_time_point = time_points[n]
        n05_time_point = (n1_time_point + n0_time_point) / 2

        # Calculate wetted length at these time points
        n1_wetted_length = np.minimum(wetted_length_wagner(n1_time_point), plate_length)
        n0_wetted_length = np.minimum(wetted_length_wagner(n0_time_point), plate_length)
        n05_wetted_length = np.minimum(wetted_length_wagner(n05_time_point), plate_length)

        # Calculate rate of change between time points (simple linear approx)
        n1_wetted_length_change = ((n1_wetted_length - n0_wetted_length)
                                   / time_step)
        n0_wetted_length_change = ((n0_wetted_length - previous_wetted_length)
                                   / time_step)
        n05_wetted_length_change = ((n05_wetted_length - n0_wetted_length)
                                    / (time_step / 2))

        # Get matrices and vectors from the generator
        matrix_1, vector_1 = matrix_vector_generator(n1_wetted_length,
                                                     n1_wetted_length_change)
        matrix_05, vector_05 = matrix_vector_generator(n05_wetted_length,
                                                       n05_wetted_length_change)
        matrix_0, vector_0 = matrix_vector_generator(n0_wetted_length,
                                                     n0_wetted_length_change)

        # Calculate cx terms for Runge Kutta
        c1_vector = time_step * matrix_vector_multiplication(matrix_0, vector_0,
                                                             current_guess)
        c2_vector = time_step * matrix_vector_multiplication(matrix_05, vector_05, current_guess + 0.5 * c1_vector)
        c3_vector = time_step * matrix_vector_multiplication(matrix_05, vector_05, current_guess + 0.5 * c2_vector)
        c4_vector = time_step * matrix_vector_multiplication(matrix_1, vector_1, current_guess + c3_vector)

        # Create a new guess {A}_i+1
        new_guess = (current_guess
                     + (c1_vector + 2 * c2_vector + 2 * c3_vector + c4_vector) / 6)

        # Save the guess
        guesses.append(float(new_guess[0]))

        # Now calculation has finished, update the current guess before
        # moving on to next iteration in the loop
        current_guess = new_guess

        # Same for the wetted length (used to calc rate of change)
        previous_wetted_length = n0_wetted_length

        # Update with text entry
        # print(f"Step {n} complete at time {n0_time_point:.7f} s. "
        #       f"Value is {float(current_guess[0]):.5e}.")

        # Exit for loop

    # Check if writing a new file is required
    if write_file:
        # Save guesses to text file
        with open('a_values.txt', 'w') as f:
            for n, value in enumerate(guesses):
                time_point = time_points[n]
                f.write(f"{value, float(time_point)}\n")


mode_one_temporal_term(write_file=True)
