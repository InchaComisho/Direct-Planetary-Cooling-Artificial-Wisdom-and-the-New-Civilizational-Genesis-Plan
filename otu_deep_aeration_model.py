"""
Base logic for a deep ocean aeration forcing model to prevent a 10% carbon uptake loss
during a super El Niño event in the Chilean-equatorial region.
"""

from dataclasses import dataclass
import numpy as np

@dataclass
class SimulationParameters:
    duration_days: int = 180
    dt_days: float = 1.0
    region_area_km2: float = 1.0e6
    mixed_layer_depth_m: float = 50.0
    deep_water_depth_m: float = 1000.0
    base_carbon_flux: float = 0.15
    super_elnino_anomaly: float = 2.5
    aeration_strength: float = 0.45
    aeration_activation_threshold: float = 1.5
    carbon_loss_target: float = 0.10


def forcing_factor(anomaly: float, threshold: float) -> float:
    """Compute a normalized forcing factor for OTU activation."""
    return max(0.0, min(1.0, (anomaly - threshold) / threshold))


def carbon_uptake_rate(base_flux: float, anomaly: float, aeration: float) -> float:
    """Return a corrected carbon uptake rate [mol/m2/day]."""
    thermal_penalty = 1.0 - 0.25 * forcing_factor(anomaly, 1.0)
    aeration_bonus = 1.0 + 0.5 * aeration
    return base_flux * thermal_penalty * aeration_bonus


def simulate_deep_aeration(params: SimulationParameters):
    days = int(params.duration_days / params.dt_days)
    time = np.linspace(0, params.duration_days, days)
    carbon_uptake = np.zeros(days)
    cumulative_loss = 0.0
    target_loss = params.carbon_loss_target

    for i in range(days):
        current_anomaly = params.super_elnino_anomaly * np.exp(-0.005 * time[i])
        control = forcing_factor(current_anomaly, params.aeration_activation_threshold)
        current_aeration = params.aeration_strength * control
        carbon_uptake[i] = carbon_uptake_rate(params.base_carbon_flux, current_anomaly, current_aeration)

        expected_no_aeration = params.base_carbon_flux * (1.0 - 0.25 * forcing_factor(current_anomaly, 1.0))
        loss_today = max(0.0, expected_no_aeration - carbon_uptake[i])
        cumulative_loss += loss_today * params.region_area_km2 * 1e6 * params.dt_days

    normalized_loss = cumulative_loss / (params.region_area_km2 * 1e6 * params.duration_days)
    return {
        "time_days": time,
        "carbon_uptake_rate": carbon_uptake,
        "cumulative_loss": cumulative_loss,
        "normalized_loss": normalized_loss,
        "target_loss": target_loss,
    }


def main():
    params = SimulationParameters()
    result = simulate_deep_aeration(params)
    print("Deep ocean aeration model base simulation")
    print(f"Duration: {params.duration_days} days")
    print(f"Super El Niño anomaly (°C): {params.super_elnino_anomaly}")
    print(f"Aeration strength: {params.aeration_strength}")
    print(f"Normalized loss estimate: {result['normalized_loss']:.4f}")
    print(f"Target carbon loss threshold: {result['target_loss']:.2f}")


if __name__ == "__main__":
    main()
