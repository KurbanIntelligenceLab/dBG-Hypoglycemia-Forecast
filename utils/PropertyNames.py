from dataclasses import dataclass


@dataclass
class ColumnNames:
    """
    A simple data class that stores the column names of the parsed dataframe.

    :ivar date: Time of the datapoint. Default is 'Date'.
    :ivar date_gap: Date gap since the last datapoint
    :ivar patient: The patient name. Default is 'Patient'.
    :ivar value: The value of the blood sugar. Default is 'Value'.
    :ivar isDangerous: Is the current value of blood suger considered dangerous?
    :ivar diff: The value change compared to the previous row. Default is 'Diff'.
    :ivar disc: The discretized range of that value. Default is 'Ranges'.
    :ivar char: The discretized representation of the value. Default is 'Char'.
    :ivar target: Target column of the alert (The ideal model)
    :ivar char_norep: Same with the char column but the subsequent repeating values are removed. Default is 'Char_No_Repeat'.
    :ivar prob_alert: Alert state using probabilistic model.
    :ivar naive_alert: Alert state using naive model.
    :ivar combined_alert_or: Alert state using combined model generated by or operation.
    :ivar combined_alert_or: Alert state using combined model generated by and operation.
    """

    date: str = 'Date'
    date_gap: str = 'Date_Gap'
    patient: str = 'Patient'
    value: str = 'Value'
    isDangerous: str = 'Dangerous'
    diff: str = 'Diff'
    disc: str = 'Ranges'
    char: str = 'Char'
    target: str = 'Target'
    char_norep: str = 'Char_No_Repeat'

    prob_alert: str = 'Probabilistic_Alert'
    naive_alert: str = 'Naive_Alert'
    combined_alert_or: str = 'Combined_Alert_Or'
    combined_alert_and: str = 'Combined_Alert_And'
    alert_columns = [prob_alert, naive_alert, combined_alert_or, combined_alert_and, target]


@dataclass
class GraphProperties:
    """
    A simple data class that stores the edge and node attribute names used in the de Bruijn graph

    :ivar weight: Weight of the edges
    :ivar tuple: The tuple subsequence that an edge represents
    """

    weight: str = 'weight'
    tuple: str = 'tuple'


@dataclass
class MethodOptions:
    """
    A simple data class that stores the names of the method options

    :ivar filter: Filter prune
    :ivar path: Path prune
    :ivar adaptive: Adaptive prune
    """

    filter: str = 'filter'
    path: str = 'path'
    adaptive: str = 'adaptive'
