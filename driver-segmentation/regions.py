from datetime import datetime
from typing import List

import db


def get_regions_to_run(type_id: int) -> List[int]:
    """Регины, для которых пора запускать сегментацию"""

    today_date = datetime.now().date()

    db_rows = db.get_mysql_conn().fetch(
        """
        SELECT t.locality_id
        FROM driver_segmentation_locality_types t
        LEFT JOIN driver_segmentation_dates d USING (type_id, locality_id)
        WHERE t.type_id = {0}
        GROUP BY t.type_id, d.locality_id
        HAVING IFNULL(MAX(d.date), "0001-01-01") < "{1}"
        ORDER BY t.type_id
        """.format(type_id, today_date.strftime("%Y-%m-%d"))
    )

    return db_rows['locality_id'].astype(int).to_list() if len(db_rows) > 0 else []
