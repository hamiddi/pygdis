# CSV example data

The file `synthetic_parameter_sweep.csv` is a simple long-format dataset for
demonstrating pyGDIS on user-supplied trajectory data.

Columns:

- `parameter`: control-parameter value identifying each trajectory;
- `time`: ordered sample index or time value;
- `x1`, `x2`: observed state variables.

For the simplest run, use the copies placed directly in the `examples` folder:

```bash
cd examples
python csv_data_example.py
```

The script creates an immediate folder named `gdis_results` beside the CSV and
writes the numerical results, two figures, and a text summary there.
