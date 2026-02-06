import pandas as pd
import ijson
import os
import geopandas as gpd
import matplotlib.pyplot as plt

class JsonToParquet:
    def convert(self, vehicle, batch_size=70000):
        input_path = f"data/raw/rea_1000m_{vehicle}_vectors_v2.json"
        output_path = f"data/raw/parquet/rea_1000m_{vehicle}_vectors_v2.parquet"
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        rows = []
        batch_num = 0
        part_files = []  # Track all created part files
        
        with open(input_path, 'r') as f:
            items = ijson.items(f, 'item')
            
            for item in items:
                origin_id = item['origin_id']
                destinations = item.get('destinations', [])
                
                # Handle case where there are no destinations
                if not destinations:
                    # Option 1: Skip origins with no destinations (current behavior)
                    # continue
                    
                    # Option 2: Keep origin with null destination (uncomment if needed)
                    # rows.append({
                    #     'origin_id': origin_id,
                    #     'destination_id': None,
                    #     'trips': None
                    # })
                    pass
                else:
                    for dest in destinations:
                        rows.append({
                            'origin_id': origin_id,
                            'destination_id': dest['destination_id'],
                            'trips': dest['trips']
                        })
                
                if len(rows) >= batch_size:
                    part_file = f"{output_path}.part{batch_num}"
                    df = pd.DataFrame(rows)
                    df.to_parquet(part_file, index=False)  # Added index=False
                    part_files.append(part_file)
                    batch_num += 1
                    rows = []
        
        # Save the final batch (if any rows remain)
        if rows:
            part_file = f"{output_path}.part{batch_num}"
            df = pd.DataFrame(rows)
            df.to_parquet(part_file, index=False)  # Added index=False
            part_files.append(part_file)
        
        print(f"{vehicle} json converted to {len(part_files)} parquet part(s)")
        
        # Merge if we have more than one part file
        if len(part_files) > 1:
            self._merge_parquet_files(output_path, part_files)
        elif len(part_files) == 1:
            # If only one part file, rename it to the final output path
            os.rename(part_files[0], output_path)
            print(f"Single part file renamed to {output_path}")
        else:
            print(f"No data processed for {vehicle}")
    
    def _merge_parquet_files(self, output_path, part_files):
        """Merge multiple parquet part files into a single file"""
        dfs = []
        
        # Read all part files
        for part_file in part_files:
            df_part = pd.read_parquet(part_file)
            dfs.append(df_part)
        
        # Concatenate all dataframes
        merged_df = pd.concat(dfs, ignore_index=True)
        
        # Write merged dataframe
        merged_df.to_parquet(output_path, index=False)
        
        # Clean up part files
        for part_file in part_files:
            if os.path.exists(part_file):
                os.remove(part_file)
        
        print(f"Merged {len(part_files)} part files into {output_path}")

def validate_parquet_conversion(vehicle):
    """
    Load and validate the converted parquet file to ensure all data was converted correctly.
    """
    output_path = f"data/raw/parquet/rea_1000m_{vehicle}_vectors_v2.parquet"
    
    try:
        # Load the parquet file
        df = pd.read_parquet(output_path)
        
        # Calculate statistics
        total_vectors = len(df)
        unique_origins = df['origin_id'].nunique()
        unique_destinations = df['destination_id'].nunique()
        total_trips = df['trips'].sum()
        
        print(f"\nValidation for {vehicle}:")
        print(f"  Total vectors (origin-destination pairs): {total_vectors:,}")
        print(f"  Unique origin IDs: {unique_origins:,}")
        print(f"  Unique destination IDs: {unique_destinations:,}")
        print(f"  Total trips: {total_trips:,}")
        
        # Optional: Show a few sample rows
        print(f"  Sample data:")
        print(df.head(3).to_string())
        
        return df
        
    except FileNotFoundError:
        print(f"Parquet file not found for {vehicle}: {output_path}")
        return None
    except Exception as e:
        print(f"Error validating {vehicle}: {e}")
        return None
    
if __name__=='__main__':
    vehhh = ['car', 'motorbike']
    toparquet = JsonToParquet()

    for v in vehhh:
        toparquet.convert(v)
        validate_parquet_conversion(v)

