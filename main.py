from utils.functions import *
import argparse
import pandas as pd
parser = argparse.ArgumentParser(description='Generate prompts for a given property file')
parser.add_argument('--maintext', type=str, help='Path to the main text file', default='documents/maintext/c60_maintext.txt')
parser.add_argument('--abstract', type=str, help='Path to the main text file',default='documents/abstract/c60_abs.txt')
parser.add_argument('--intro', type=str, help='Path to the main text file', default='documents/intro/c60_intro.txt')
parser.add_argument('--window_size', type=int, help='Window size for the sliding window during reading the main text', default=2)
parser.add_argument('--chunk_size', type=int, help='Chunk size for the text splitter', default=1000)
parser.add_argument('--chunk_overlap', type=int, help='Overlap size for the text splitter', default=100)
parser.add_argument('--output', type=str, help='Output file for the prompts', default='results')
parser.add_argument('--property', type=str, help='Property to generate prompts for', default='prompts/factors.txt')
parser.add_argument('--self_defined', type=bool, help='Whether to use self defined properties', default=True)
parser.add_argument('--title', type=str, help='Title of the paper', required=True)


def main():
    args = parser.parse_args()
    print(args)

    # firstly check if this has been parsed before
 
    # Define the path to your CSV file
    csv_path = "documents/parsed.csv"

    # Check if the file exists
    if not os.path.exists(csv_path):
        # If the file doesn't exist, create it with the specified headers
        df = pd.DataFrame(columns=['title'])
        df.to_csv(csv_path, index=False)
    else:
        # If the file exists, read it
        df = pd.read_csv(csv_path)
    visited = set(df['title'].tolist())
    if args.title in visited:
        print("This paper has been parsed before")
        #return



    # Load the main text
    self_defined_properties = read_properties(args.property)
    main_text = read_file(args.maintext)
    abstract = read_file(args.abstract)
    intro = read_file(args.intro)

    main_material = ""
    factors = []
    

    # extract main material name and chemical symbol
    main_topic_response = extract_main_topic(abstract)
    main_material = main_topic_response.main_material_name + " " + main_topic_response.chemical_symbol


    # extract factors
    #factors = extract_factors(abstract, intro)
    if args.self_defined:
        factors.extend(self_defined_properties)   
    
    # # demonstration only
    # main_material = 'C60-SWCNT electrocatalyst' 
    # factors = ['adsorption of C60 onto SWCNTs',' enhanced electrocatalytic activities', 'heteroatom doping']

    # extract property data from main text
    results, indexed_text = extract_properties(main_text, factors, main_material, args.window_size, args.chunk_size, args.chunk_overlap)
    
    # write json to args.output (with timestamp encoded name)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = args.output+'/'+main_material.replace(" ", "_").replace("/", "_") + '_result_' +timestamp+".json"
    text_filename = args.output+'/'+main_material.replace(" ", "_").replace("/", "_") + '_text_'+timestamp+'.txt' 
    results.meta = parser.parse_args().__dict__
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(results.json(indent=2))
        f.write('\n')
        #json.dump(parser.parse_args().__dict__, f,ensure_ascii=False,  indent=2)

    with open(text_filename, 'w', encoding='utf-8') as f:
        json.dump(indexed_text, f, ensure_ascii=False, indent=2)
        

if __name__ == "__main__":
    main()
