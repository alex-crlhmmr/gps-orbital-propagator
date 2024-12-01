from tle_utils import process_tle









if __name__ == '__main__':
    orbital_data = process_tle('tle.txt')
    print(orbital_data)