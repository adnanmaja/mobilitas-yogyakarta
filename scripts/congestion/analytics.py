from typing import Dict, List, Tuple, Optional
from congestion_config import Config

class Analytics:
    def __init__(self):
        pass

    def calculate_statistics(self, edges: List[Dict]):
        """ Calculate and print summary statistics including mean and median travel times. """
        import statistics

        car_times = []
        bike_times = []
        
        # For flow-weighted averages
        total_car_travel_time = 0
        total_car_volume = 0
        total_bike_travel_time = 0
        total_bike_volume = 0

        congested_segments = 0
        total_segments = len(edges)
        
        for edge in edges:
            props = edge['properties']
            c_time = props.get('car_travel_time', 0)
            b_time = props.get('motorbike_travel_time', 0)
            c_flow = props.get('car_flow', 0)
            b_flow = props.get('motorbike_flow', 0)

            # Collect for Median and Simple Mean
            if c_time > 0: car_times.append(c_time)
            if b_time > 0: bike_times.append(b_time)

            # Collect for Flow-Weighted Mean
            total_car_travel_time += (c_time * c_flow)
            total_car_volume += c_flow
            total_bike_travel_time += (b_time * b_flow)
            total_bike_volume += b_flow
            
            if props.get('vc_ratio', 0) > 0.8:
                congested_segments += 1
        
        # Calculate Stats
        avg_car = statistics.mean(car_times) if car_times else 0
        med_car = statistics.median(car_times) if car_times else 0
        avg_bike = statistics.mean(bike_times) if bike_times else 0
        med_bike = statistics.median(bike_times) if bike_times else 0
        
        weighted_car = total_car_travel_time / total_car_volume if total_car_volume > 0 else 0
        weighted_bike = total_bike_travel_time / total_bike_volume if total_bike_volume > 0 else 0

        print("\n" + "="*50)
        print("NETWORK STATISTICS")
        print("="*50)
        print(f"CONGESTION: {congested_segments}/{total_segments} segments at v/c > 0.8")
        
        print("\nCAR TRAVEL TIMES (seconds per segment):")
        print(f"  Average: {avg_car:.2f}s")
        print(f"  Median:  {med_car:.2f}s")
        print(f"  Weighted Average: {weighted_car:.2f}s (based on flow)")

        print("\nMOTORBIKE TRAVEL TIMES (seconds per segment):")
        print(f"  Average: {avg_bike:.2f}s")
        print(f"  Median:  {med_bike:.2f}s")
        print(f"  Weighted Average: {weighted_bike:.2f}s (based on flow)")
        print("="*50)