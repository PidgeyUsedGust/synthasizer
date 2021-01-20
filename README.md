# Synthasizer

A tool for semi-automated wrangling of tables into attribute-value format.

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
  * For other columns, we either compute similarity to the colored cell or average similarity across all cells if no color is available.
  
  We are continuously looking for spreadsheets and cases in which these rules are violated in order to finetune the heuristics.
  
* A new **reconstruction error** uses heuristics that depend on different cases, such as the tables changing in number of rows, column or not changing in size at all. 

## TODO

* Implement new reconstruction error.
* Implement search algorithm

## Future improvements

* Add support for **FlashProfile**. For example, the `Split` transformation could also use FlashProfile derived splits.
* Add support for using cell style in heuristics.
