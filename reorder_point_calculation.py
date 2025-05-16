import math

class ReorderPointCalculator:
    def __init__(self, z_score=1.65):  # 95% service level
        self.z_score = z_score

    def calculate(self, avg_daily_demand, lead_time, demand_std_dev=None):
        if demand_std_dev is None:
            demand_std_dev = avg_daily_demand * 0.3  # Default to 30% variability

        lead_time_demand = avg_daily_demand * lead_time
        safety_stock = self.z_score * demand_std_dev * math.sqrt(lead_time)
        reorder_point = lead_time_demand + safety_stock

        return {
            "avg_daily_demand": avg_daily_demand,
            "lead_time": lead_time,
            "std_dev": demand_std_dev,
            "lead_time_demand": round(lead_time_demand, 2),
            "safety_stock": round(safety_stock, 2),
            "reorder_point": round(reorder_point, 2)
        }
