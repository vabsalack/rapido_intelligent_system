1. Bookings CSV
- Memory usage: 16.8 MB
- Total rows: 1 lakh
- Total columns (without index): 22
- dtypes: float64(6), int64(2), str(14)
| Features                  | Type    | Stat Nature       | Range            | Nulls? | Is target? |
| ------------------------- | ------- | ----------------- | ---------------- | ------ | ---------- |
| "booking_id"              | str     | identifier        | All unique       | no     |            |
| "customer_id"             | str     | identifier        | Not unique       | no     |            |
| "driver_id"               | str     | identifier        | Not unique       | no     |            |
|                           |         |                   |                  |        |            |
| "is_weekend"              | int64   | discrete ratio    | 0 / 1            | no     |            |
| "hour_of_day"             | int64   | discrete ratio    | [0, 23]          | no     |            |
| "ride_distance_km"        | float64 | continuous ratio  | [1, 25]          | no     |            |
| "estimated_ride_time_min" | float64 | continuous ratio  | [3, 164.98]      | no     |            |
| "actual_ride_time_min"    | float64 | continuous ratio  | [2.75, 197.34]   | 31654  |            |
| "base_fare"               | float64 | continuous ratio  | [28.02, 529.96]  | no     |            |
| "surge_multiplier"        | float64 | continuous ratio  | [1, 2.3]         | no     |            |
| "booking_value"           | float64 | continuous ratio  | [27.28, 1265.59] | no     | yes R      |
|                           |         |                   |                  |        |            |
| "booking_date"            | str     | YYYY:MM:DD        | 365              | no     |            |
| "booking_time"            | str     | HH:MM:SS          | 1440             | no     |            |
| "day_of_week"             | str     | nominal / ordinal | 7                | no     |            |
| "city"                    | str     | nomial            | 5                | no     |            |
| "pickup_location"         | str     | nominal           | 50               | no     |            |
| "drop_location"           | str     | nominal           | 50               | no     |            |
| "vehicle_type"            | str     | nominal / ordinal | 3                | no     |            |
| "traffic_level"           | str     | ordinal           | 3                | no     |            |
| "weather_condition"       | str     | oridnal           | 3                | no     |            |
| "booking_status"          | str     | nominal           | 3                | no     | yes L=3    |
| "incomplete_ride_reason"  | str     | nominal           | 5                | 91630  |            

2. Customers CSV
- Memory usage: 0.99 MB
- Total rows: 10 K
- Total columns: 13
- dtypes: float64(2), int64(7), str(4)
| Features                   | Type    | Stat Nature       | Range      | Nulls? | Is target? |
| -------------------------- | ------- | ----------------- | ---------- | ------ | ---------- |
| "customer_id"              | str     | identifier        | All Unique | no     |            |
|                            |         |                   |            |        |            |
| "customer_age"             | int64   | discrete ratio    | [18, 64]   | no     |            |
| "customer_signup_days_ago" | int64   | discrete ratio    | [30, 999]  | no     |            |
| "total_bookings"           | int64   | discrete ratio    | [1, 26]    | no     |            |
| "completed_rides"          | int64   | discrete ratio    | [0, 20]    | no     |            |
| "cancelled_rides"          | int64   | discrete ratio    | [0, 10]    | no     |            |
| "incomplete_rides"         | int64   | discrete ratio    | [0, 5]     | no     |            |
| "cancellation_rate"        | float64 | continuous ratio  | [0.0, 1.0] | no     |            |
| "avg_customer_rating"      | float64 | continuous ratio  | [0.0, 5.0] | no     |            |
| "customer_cancel_flag"     | int64   | discrete ratio    | 0 / 1      | no     | yes L = 2  |
|                            |         |                   |            |        |            |
| "customer_gender"          | str     | nominal           | 3          | no     |            |
| "customer_city"            | str     | nominal           | 5          | no     |            |
| "preferred_vehicle_type"   | str     | nominal / ordinal | 3          | no     |            |

3. Drivers CSV
- Memory Usage: 0.53 MB
- Total rows: 5 K
- Total columns: 14
- dtypes: float64(4), int64(7), str(3)
| Features                  | Type    | Stat Nature       | Range         | Nulls? | Is target? |
| ------------------------- | ------- | ----------------- | ------------- | ------ | ---------- |
| "driver_id"               | str     | identifier        | All unique    | no     |            |
|                           |         |                   |               |        |            |
| "driver_age"              | int64   | discrete ratio    | [22, 54]      | no     |            |
| "driver_experience_years" | int64   | discrete ratio    | [1, 14]       | no     |            |
| "total_assigned_rides"    | int64   | discrete ratio    | [6, 38]       | no     |            |
| "accepted_rides"          | int64   | discrete ratio    | [3, 30]       | no     |            |
| "incomplete_rides"        | int64   | discrete ratio    | [0, 7]        | no     |            |
| "delay_count"             | int64   | discrete ratio    | [0, 6]        | no     |            |
| "driver_delay_flag"       | int64   | discrete ratio    | 0 / 1         | no     | yes L = 2  |
| "acceptance_rate"         | float64 | continuous ratio  | [0.31, 1.00]  | no     |            |
| "delay_rate"              | float64 | continuous ratio  | [0.00, 0.42]  | no     |            |
| "avg_driver_rating"       | float64 | continuous ratio  | [4.00, 5.00]  | no     |            |
| "avg_pickup_delay_min"    | float64 | continuous ratio  | [1.00, 10.30] | no     |            |
|                           |         |                   |               |        |            |
| "driver_city"             | str     | nominal           | 5             | no     |            |
| "vehicle_type"            | str     | nominal / ordinal | 3             | no     |            |

4. Location Demand CSV
- Memory usage: 1.4 MB
- Total rows: 17,941 $\simeq$ 18 K
- Total columns: 10
- dtypes: float64(2), int64(4), str(4)
| Features               | Type    | Stat Nature       | Range          | Nulls? | Is target? |
| ---------------------- | ------- | ----------------- | -------------- | ------ | ---------- |
| No Ids                 |         |                   |                |        |            |
|                        |         |                   |                |        |            |
| "hour_of_day"          | int64   | discrete ratio    | [0, 23]        | no     |            |
| "total_requests"       | int64   | discrete ratio    | [1, 15]        | no     |            |
| "completed_rides"      | int64   | discrete ratio    | [0, 13]        | no     |            |
| "cancelled_rides"      | int64   | discrete ratio    | [0, 8]         | no     |            |
| "avg_wait_time_min"    | float64 | continuous ratio  | [3.74, 164.19] | no     |            |
| "avg_surge_multiplier" | float64 | continuous ratio  | [1.0, 2.30]    | no     |            |
|                        |         |                   |                |        |            |
| "city"                 | str     | nominal           | 5              | no     |            |
| "pickup_location"      | str     | nominal           | 50             | no     |            |
| "vehicle_type"         | str     | nominal / ordinal | 3              | no     |            |
| "demand_level"         | str     | ordinal           | 2              | no     |            |

5. Time Features CSV
- Memory usage: 0.46 MB
- Total rows: 8760 $\simeq$ 9 K
- Total columns: 7
- dtypes: int64(4), str(3)
| Features         | Type  | Stat Nature           | Range   | Nulls? | Is target? |
| ---------------- | ----- | --------------------- | ------- | ------ | ---------- |
| No Ids           |       |                       |         |        |            |
|                  |       |                       |         |        |            |
| "hour_of_day"    | int64 | discrete ratio        | [0, 23] | no     |            |
| "is_weekend"     | int64 | discrete ratio        | 0 / 1   | no     |            |
| "is_holiday"     | int64 | discrete ratio        | only 0s | no     |            |
| "peak_time_flag" | int64 | discrete ratio        | 0 / 1   | no     |            |
|                  |       |                       |         |        |            |
| "datetime"       | str   | 'YYYY:MM:DD HH:MM:SS' | 8,760   | no     |            |
| "day_of_week"    | str   | nominal / ordinal     | 7       | no     |            |
| "season"         | str   | ordinal / nominal     | 3       | no     |            |

