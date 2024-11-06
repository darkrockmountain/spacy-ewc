from typing import Dict
from thinc.types import FloatsXd
import numpy as np


class VectorDict(Dict[str, FloatsXd]):
    """
    VectorDict is a specialized dictionary subclass that maps string keys to
    multidimensional floating-point arrays (FloatsXd). It provides customized
    string representations for easy readability when working with large arrays.

    This class is typically used to store and display model parameters or gradient
    information in an organized manner.
    """

    def __str__(self) -> str:
        """
        Provide a readable string representation of the dictionary, displaying
        truncated array contents for each key-value pair. Each key and its
        corresponding array are shown on a new line.

        Returns:
            str: A formatted string where each line contains a key followed by
                 a truncated view of its array contents.
        """
        if not self:
            # Return an empty dictionary representation if there are no items.
            return "{}"
        # Generate a formatted string for each key-value pair in the
        # dictionary.
        return "\n".join(
            [f"{key}: {self._format_array(value)}" for key, value in self.items()]
        )

    def __repr__(self) -> str:
        """
        Provide a detailed representation of the VectorDict, suitable for debugging.
        This includes the dictionary name and truncated array contents for each key-value pair.

        Returns:
            str: A formatted string representing the dictionary in a debug-friendly format.
        """
        # Generate a formatted string for each key-value pair in the dictionary
        # and include the class name for clarity in debugging.
        return f"VectorDict({{ {', '.join(f'{key}: {self._format_array(value)}' for key, value in self.items())} }})"

    @staticmethod
    def _format_array(arr: np.ndarray, num_elements: int = 5) -> str:
        """
        Format an array for display by flattening it and limiting the number of
        displayed elements. This helps make large arrays readable in summary form.

        Args:
            arr (np.ndarray): The array to be formatted.
            num_elements (int, optional): The maximum number of elements to display.
                                          Default is 5.

        Returns:
            str: A truncated list representation of the array followed by ellipses
                 if there are additional elements.
        """
        # Flatten the array to 1D and select up to the first `num_elements`
        # items.
        flat_arr = arr.ravel()[:num_elements]
        # Convert the selected portion to a list and add ellipses for
        # truncation.
        return f"{flat_arr.tolist()}..."
