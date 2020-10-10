import pymongo
import pymysql


class Mongo2mysql:
    SYSTEM = 'sys'

    def __init__(self):
        # connect mongo
        client = pymongo.MongoClient('localhost', 27017)
        self.mongoDB = client['maas']
        # connect mysql
        self.mysql = pymysql.connect(host='localhost', user='root', password='admin123', port=3306)

    """
    CREATE TABLE `t_placeholder` (
      `pid` int(11) unsigned NOT NULL AUTO_INCREMENT,
      `id` varchar(24) NOT NULL DEFAULT '',
      `name` varchar(128) DEFAULT NULL,
      `type` varchar(16) DEFAULT NULL,
      `score_name` varchar(256) DEFAULT NULL,
      `service_name` varchar(128) DEFAULT NULL,
      `pool_name` varchar(128) DEFAULT NULL,
      `create_time` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
      `update_time` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
      `create_by` varchar(64) DEFAULT NULL,
      `update_by` varchar(64) DEFAULT NULL,
      PRIMARY KEY (`pid`),
      KEY `IX_id` (`id`),
      KEY `IX_name` (`name`),
      KEY `IX_create_time` (`create_time`),
      KEY `IX_updateTime` (`update_time`)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """
    def migration_data(self):

        # get table
        table = self.mongoDB['placeholder']
        cursor = self.mysql.cursor()

        # read and insert into mysql
        for x in table.find():
            print(x)
            sql = 'insert into maas.t_placeholder' \
                  '(id,name,type,score_name,service_name,pool_name,create_time,update_time,create_by,update_by) ' \
                  'values("%s","%s","%s","%s","%s","%s","%s","%s","%s","%s")' % \
                  (self.get(x, '_id'), self.get(x, 'name'), self.get(x, 'type'),
                   self.get(x, 'scoreName'), self.get(x, 'serviceName'), self.get(x, 'poolName'),
                   self.get(x, 'createTime'), self.get(x, 'createTime'), self.SYSTEM, self.SYSTEM)
            print(sql)
            try:
                cursor.execute(sql)
                self.mysql.commit()
            except BaseException as e:
                print(e)
                self.mysql.rollback()

    @classmethod
    def get(cls, data, field):
        if '_id' == field:
            return str(data.get('_id'))
        if data.get(field) is None:
            return ''
        return data.get(field)


def main():
    Mongo2mysql().migration_data()


if __name__ == "__main__":
    main()
