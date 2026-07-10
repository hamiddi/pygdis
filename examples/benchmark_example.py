from pathlib import Path
from gdis import GDIS
from gdis.plotting import plot_gdis
from gdis.systems import ChenSystem, LogisticMap, LorenzSystem, RosslerSystem

output = Path("benchmark_results")
output.mkdir(exist_ok=True)

for system in [LorenzSystem(), RosslerSystem(), ChenSystem(), LogisticMap()]:
    trajectories, parameters = system.generate_sweep()
    result = GDIS().fit_transform(
        trajectories,
        parameters,
        jacobian_function=getattr(system, "jacobian", None),
        critical_value=system.critical_value,
    )
    system_dir = output / system.name.lower()
    system_dir.mkdir(exist_ok=True)
    result.to_dataframe().to_csv(system_dir / "results.csv", index=False)
    plot_gdis(result, output_path=str(system_dir / "gdis.png"))
