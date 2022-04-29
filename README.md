# Synthasizer

A tool for semi-automated wrangling of tables into attribute-value format. Significantly improved version of the algorithm described in [this paper](https://github.com/PidgeyUsedGust/synthasizer/blob/main/synthasizer_ida18.pdf) (see below for differences).

*(This repository contains the basis for some experimental features, such as learning wrangling patterns and phased wrangling. These will be finalized and described soon.)*

## Usage

First, we load a table, either from a `pd.DataFrame`, a csv file, or a slice of an openpyxl `Worksheet`. (There is also a convenience method `detect` for detecting table boundaries in worksheets.)

```python
> sheet = openpyxl.load_workbook("tests/data/icecream.xlsx")["month"]
> table = Table.from_openpyxl(sheet["A1:G10"])
> table

           0        1    2    3    4      5       6
0       Type  Country  Jun  Jul  Aug  Total  Profit
1     Banana       BE  170  690  520   1380     YES
2                  DE  610  640  320   1570      NO
3                  DE  250  650  630   1530     YES
4  Chocolate       BE  560  320  140   1020     YES
5                  FR  430  350  300   1080     YES
6                                                  
7                  NL  210  280  270    760      NO
8  Speculaas       BE  300  270  290    860      NO
9    Vanilla       BE  610  190  670   1470     YES
```

We can manually write a transformation program to fix it.

```python
> wrangled = Program([Fill(0), Delete(1, EmptyCondition())])(table)
> wrangled

           0        1    2    3    4      5       6
0       Type  Country  Jun  Jul  Aug  Total  Profit
1     Banana       BE  170  690  520   1380     YES
2     Banana       DE  610  640  320   1570      NO
3     Banana       DE  250  650  630   1530     YES
4  Chocolate       BE  560  320  140   1020     YES
5  Chocolate       FR  430  350  300   1080     YES
6  Chocolate       NL  210  280  270    760      NO
7  Speculaas       BE  300  270  290    860      NO
8    Vanilla       BE  610  190  670   1470     YES
```

Or we can learn this program automatically.

```python
# a phase of wrangling consists of a language and a heuristic
phase = (
    Language([Delete, Divide, Header, Fill, Stack, Fold]),
    WeightedHeuristic(
        [
            EmptyColumnHeuristic(),
            EmptyRowHeuristic(),
            TypeHeuristic()
        ],
        weights=[0.2, 0.2, 1],
    )
)
wrangler = Wrangler([phase])
programs = wrangler.learn(table)
```

Multiple programs can be found, but the first one will be the most likely one (and in this case the only one).

```python
> programs[0].Program
Header(1), Fill(0), Delete(1, EmptyCondition)
```

### Strategies and Heuristics

Different strategies and heuristics are provided. We are working on selecting good defaults and will update the API accordingly.

### Optimisations

Some optimizations can be set.

```python
from settings import Optimizations

# does not use cell style when determining transformations
Optimizations.disable_style()

# does not automatically fold columns with headers that
# we think should be folded, like weekdays, months and
# increasing sequences of integers
Optimizations.disable_smart()
```

## Differences with respect to paper

Some improvements have been made over the years to make it more robust for real spreadsheets.

* Transformations can use **cell style** in both their working and syntactic bias. Some transformations that use this are `Delete`.

* Syntactic bias and heuristics now also look at cell **types**. They are the types from the [`infer_dtypes`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.api.types.infer_dtype.html) function.

* Two **new transformations** are added.
  * A `Header` transformation is added that sets the first `n` rows as column headers. Headers are not used for the similarity computation.
  * A `Split` function has been added that uses cell style for defining splitting criteria. It allows to split on any stylistic property.
  
* The **heuristic** now combines column- and row based heuristics in a more sensible way.
  * Columns that contain multiple colors don't get points.
  * Similarities are only computed between columns that are **strings** or **mixed**. For all other types, we assume that a clean column is already correct.
  * String similarity is included, but not used by default.
  
  We are continuously looking for spreadsheets and cases in which these rules are violated in order to finetune the heuristics.
  
* A new **reconstruction error** uses heuristics that depend on different cases, such as the tables changing in number of rows, column or not changing in size at all. 


## TODO

* Add pattern learning and include a set of default patterns.
* Update the API with good defaults.
* Add support for **FlashProfile**. For example, the `Split` transformation could also use FlashProfile derived splits.
