from pytest import mark
from synthasizer.transformation import Delete


@mark.parametrize("car", [False], indirect=True)
def test_delete(car):
    # print(car)
    print(car.color_df)
    print(car.color_foreground().color_df)
    print(car.color_df)