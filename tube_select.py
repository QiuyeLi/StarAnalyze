import argparse

################################################################################################
#   python tube_select.py --i particles.star all_particles.star --o selected_particles.star    #
################################################################################################ 

def find_indices(filename):
    """Find indices of _rlnHelicalTubeID # and _rlnMicrographName #"""
    a_idx, b_idx = -1, -1
    with open(filename, 'r') as file:
        for line in file:
            if line.startswith('_rlnHelicalTubeID #'):
                a_idx = int(line.split('#')[1].strip()) - 1
                # Print 1-based index for debugging
                print(f"Found _rlnHelicalTubeID # at index: {a_idx + 1}")
            elif line.startswith('_rlnMicrographName #'):
                b_idx = int(line.split('#')[1].strip()) - 1
                # Print 1-based index for debugging
                print(f"Found _rlnMicrographName # at index: {b_idx + 1}")
            # Exit loop once both indices are found
            if a_idx != -1 and b_idx != -1:
                break
    
    return a_idx, b_idx


def read_star_file(filename):
    """Read data from a STAR file, returning the header and data lines."""
    header = []
    data = []
    data_start_index = None

    with open(filename, 'r') as file:
        lines = file.readlines()

    # Find the start of the data block after the last occurrence of "loop_"
    last_loop_index = None
    for i, line in enumerate(lines):
        if line.strip() == 'loop_':
            last_loop_index = i

    # Find the start of the data block (first line after the last "loop_" that doesn't start with "_rln")
    if last_loop_index is not None:
        for i in range(last_loop_index + 1, len(lines)):
            line = lines[i].strip()
            if not line.startswith('_rln'):
                data_start_index = i
                break

    # Separate the header and data blocks
    if data_start_index is not None:
        header = lines[:data_start_index]
        data = lines[data_start_index:]  # Data block extends to the end of the file

        # Clean up data
        data = [line.strip().split() for line in data if line.strip()]

    print(f"Header length: {len(header)} lines")
    print(f"Data length: {len(data)} lines")

    return header, data




def filter_data(particle_data, all_particle_data, a_idx, b_idx, c_idx, d_idx):
    """Filter all_particle_data based on keys from particle_data."""
    particle_dict = {}
    for line in particle_data:
        if len(line) > max(a_idx, b_idx):
            key = (line[a_idx], line[b_idx])
            particle_dict[key] = line

    new_particle_data = []
    for line in all_particle_data:
        if len(line) > max(c_idx, d_idx):
            key = (line[c_idx], line[d_idx])
            if key in particle_dict:
                new_particle_data.append(line)

    print(f"Filtered data length: {len(new_particle_data)}")
    return new_particle_data

def write_star_file(filename, header, data):
    """Write data to a STAR file with the given header and data."""
    with open(filename, 'w') as file:
        # Write header
        for line in header:
            file.write(line)
        
        # Write data
        for line in data:
            file.write('\t'.join(line) + '\n')

def main():
    parser = argparse.ArgumentParser(description='Filter all_particles.star based on particles.star')
    parser.add_argument('--i', nargs=2, required=True, help='Input STAR files: particles.star and all_particles.star')
    parser.add_argument('--o', required=True, help='Output STAR file: selected_particles.star')
    args = parser.parse_args()

    particles_file = args.i[0]
    all_particles_file = args.i[1]
    output_file = args.o

    # Find indices in particles.star and all_particles.star
    a_idx, b_idx = find_indices(particles_file)
    c_idx, d_idx = find_indices(all_particles_file)

    # Read header and data from files
    particle_header, particle_data = read_star_file(particles_file)
    all_particle_header, all_particle_data = read_star_file(all_particles_file)

    # Filter data
    new_particle_data = filter_data(particle_data, all_particle_data, a_idx, b_idx, c_idx, d_idx)

    # Write new particle data to the output file with the full header from all_particles.star
    write_star_file(output_file, all_particle_header, new_particle_data)

if __name__ == '__main__':
    main()
