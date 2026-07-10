from gdis import GDIS
from gdis.plotting import plot_components, plot_gdis
from gdis.systems import LorenzSystem

system = LorenzSystem()
trajectories, parameters = system.generate_sweep()
result = GDIS().fit_transform(
    trajectories,
    parameters,
    jacobian_function=system.jacobian,
    critical_value=system.critical_value,
)
result.to_dataframe().to_csv("lorenz_gdis_results.csv", index=False)
plot_gdis(result, output_path="lorenz_gdis.png")
plot_components(result, output_path="lorenz_components.png")
