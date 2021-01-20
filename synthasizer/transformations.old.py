"""Transformations to be used for wrangling."""
import pandas as pd
import numpy as np


class Transformation:
    """Abstract transformation class."""

    def __init__(self):
        """Initialise transformation."""
        pass

    def apply(self, table):
        """Apply transformation to a table."""
        pass

    @classmethod
    def arguments(table):
        """Valid arguments of transformation on table.

        Returns:
            A tuple or arguments `a` such that `Transformation(*a)` can be used to
            initialise a transformation that works on `table`.

        """
        pass

    @classmethod
    def arguments_heuristic(table):
        """Valid arguments of transformation on table and their heuristic.

        Arguments:
            table (pd.DataFrame): Segmented table in which each segment should
                (approximately) contain elements of the same data stype, i.e.,
                values that should end up in the same column.

        Returns:
            A list of tuple `(a, h)` such that `Transformation(*a)` can be used to
            initialise a transformation that works on `table` and h the heuristic
            of these arguments.

        """
        pass

    def __str__(self):
        return "{}()".format(self.__class__.__name__)


class Fill(Transformation):
    """Fill a table column.
    
    Two modes are available: forward and backward.   

    """

    modes = ["back", "forth"]
    """Filling modes."""

    def __init__(self, column, mode):
        """Initialise Fill.
        
        Args:
            column (int): Column to fill.
            mode (str): Filling mode, either 'back' or 'forth'.

        """
        self.column = column

    def apply(self, table, segmentation=None):
        """Fill a table."""

        # fill
        table.iloc[:, self.column].ffill(inplace=True)

        # return with segmentation
        if segmentation:
            return table, segmentation

        # no segmentation
        return table

    @classmethod
    def arguments(cls, table):
        """Possible fill arguments.
        
        There needs to be some empty values. 

        """
        arguments = list()
        for i in range(table.shape[1]):
            if table.iloc[:, i].isnull().any():
                arguments.append((i,))
        return arguments

    @classmethod
    def arguments_segmentation(cls, table, segmentation):
        """Possible arguments with segmentation.
        
        All affected cells should be in the same segment.

        """
        arguments = list()
        for i in range(table.shape[1]):
            # empty values
            if table.iloc[:, i].isnull().any():
                # get column
                sc = table.iloc[:, i]
                # get index of first element that is affected
                first = (sc.notnull().shift(1) & sc.isnull()).idxmax() - 1
                # no elements affected, not valid
                if first < 0:
                    continue
                # get index of last element that is affected
                last = (
                    table.shape[0]
                    + 2
                    - (sc.isnull().shift(1) & sc.notnull())[::-1].idxmax()
                )
                # check if they are in same segment
                if segmentation.iloc[first:last, i].nunique() == 1:
                    arguments.append((i,))
        return arguments


class Fold:
    """Fold columns.
    
    For now, we limit to single-level folds.

    """

    def __init__(self, c1, c2, level=1):
        """Fold."""
        self.c1 = c1
        self.c2 = c2
        self.level = level

    def apply(self, table, segmentation=None):
        """Apply fold.
        
        Segments are (luckily) quite easy.

        """

        # make first levels as header and add a dummy row
        # so that duplicate column names are allowed.
        table = (
            table.T.reset_index(level=0)
            .set_index(["index"] + list(range(self.level)))
            .T
        )
        # identifier variables are everything but the melted columns
        id_vars = (
            table.columns[: self.c1].tolist() + table.columns[self.c2 + 1 :].tolist()
        )
        # melt
        table = table.melt(id_vars=id_vars).drop(columns="index")
        # move melted columns back in place
        columns = table.columns.tolist()
        columns = (
            columns[: self.c1]
            + columns[-(self.level + 1) :]
            + columns[self.c1 : -(self.level + 1)]
        )
        table = table[columns]
        # get original index
        header = table.columns.tolist()
        header[self.c1 : self.c1 + self.level + 1] = [(np.nan,)] * (self.level + 1)
        header = pd.DataFrame(header).T.fillna(value=pd.np.nan)
        # reset table index and concatenate melted table with
        # original index
        table.columns = header.columns
        table = pd.concat((header.iloc[1:], table)).reset_index(drop=True)

        # segmentation
        if segmentation is not None:
            # creating from scratch will be easier
            new = pd.DataFrame(0, index=table.index, columns=table.columns)
            # number of columns folded
            d = self.c2 - self.c1 + 1
            # before folded, values and header
            new.iloc[self.level :, : self.c1] = pd.concat(
                [segmentation.iloc[self.level :, : self.c1]] * d
            ).values
            new.iloc[: self.level, : self.c1] = segmentation.iloc[
                : self.level, : self.c1
            ]
            # after folded, values and header
            new.iloc[self.level :, self.c1 + self.level + 1 :] = pd.concat(
                [segmentation.iloc[self.level :, self.c2 + 1 :]] * d
            ).values
            new.iloc[: self.level, self.c1 + self.level + 1 :] = segmentation.iloc[
                : self.level, self.c2 + 1 :
            ]
            # folded, we assume the headers and data to
            # be properly segmented
            for i in range(self.level + 1):
                new.iloc[self.level :, self.c1 + i] = segmentation.iloc[i, self.c1]
            # new headers are nan
            new.iloc[: self.level, self.c1 : self.c1 + self.level + 1] = pd.np.nan
            
            # return table and new segmentation
            return table, new

        return table

    @classmethod
    def arguments(cls, table):
        pass

    @classmethod
    def arguments_segmentation(cls, table, segmentation):
        pass


class Divide:
    """Divide a column in multiple columns.
    
    Divided either by a pattern (as a regular expression) or a mask, for
    example a segmentation.
    
    """

    def __init__(self, column, on=None):
        """Initialise split.
        
        Args:
            column (int): Column to split.
            on (str): String to split on.
        """
        self.column = column
        if on is None:
            on = [1]
        self.on = np.array(on)

    def apply(self, table):
        # get mask dimensions
        # on = on *

        for i, v in enumerate(np.unique(self.on)):
            # insert new column
            table.insert(
                self.column + i + 1,
                column="{}_{}".format(table.columns[i] + 1, i),
                value=table.iloc[:, self.column],
                allow_duplicates=True,
            )
            # remove everything that is not this mask
            # print(table.iloc[:, self.column + i + 1])
            print(self.on != i)
        return table


class Merge:
    """Merge columns together."""

    def __init__(self, columns):
        """
        
        Args:
            columns (tuple): Tuple of (first, last) column to merge.
    
        """
        self.i, self.j = columns

    def apply(self, table):
        # compute new column
        new_column = (
            table.iloc[:, self.i : self.j + 1]
            .replace(np.nan, "")
            .apply(lambda row: "".join(row.values.astype(str)), axis=1)
        )
