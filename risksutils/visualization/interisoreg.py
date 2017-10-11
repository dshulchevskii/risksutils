from collections import namedtuple
import holoviews as hv
import numpy as np
import pandas as pd
from ._static_plot import isotonic


Plot = namedtuple('Plot', ['selector', 'diagram'])


class InterIsoReg():
    """Интерактивная визуализация точности прогноза вероятности"""

    def __init__(self, data, pdims, tdims, ddims=None, gdims=None,
                 calibrations_data=None):
        """Интерактивная визуализация точности прогноза вероятности

        Аргументы:
          data: pandas.DataFrame
            таблица с данными
          pdims: list
            список названий столбцов с предсказаниями
          tdims: list
            целевые переменные
          ddims: list
            даты
          gdims: list
            категорияльные поля

        Результат:
            InterIsoReg - объект с набором
            интерактивных диаграмм
        """
        self.data = data
        self._pdims = pdims
        self._tdims = tdims
        self._gdims = gdims
        self._ddims = ddims
        self._calibrations_data = calibrations_data
        self._check_fields()        # Проверяем форматы полей
        self._load_calibrations()   # загружаем калибровки, если они есть
        self._diagrams = {}         # Здесь будем хранить диаграммы
        self._make_bars_static()    # Создаем диаграммы с категориями
        self._make_area_static()    # С датами
        self._make_charts()         # Конвертим их в готовые диаграммы
        self._make_isotinic_plot()

    def _get_count(self, dim, conditions=None):

        if conditions is None:
            df = self.data
        else:
            df = self.data.loc[conditions]

        return (df.groupby(dim)
                .size()
                .reset_index()
                .rename(columns={0: 'count'}))

    def _make_bars_static(self):
        """Создаем столбчатые диаграммы с выбором категорий"""
        for dim in self._gdims:
            df = self._get_count(dim)
            diagram = (hv.Bars(df, kdims=[dim], vdims=['count'])
                       .opts(plot=dict(tools=['tap'])))
            selector = (hv.streams
                        .Selection1D(source=diagram)
                        .rename(index=dim))
            self._diagrams[dim] = Plot(selector, diagram)

    def _make_area_static(self):
        """Создаем диаграммы с выбором диапозона дат"""
        for dim in self._ddims:
            df = self._get_count(dim)
            diagram = hv.Area(df, kdims=[dim], vdims=['count'])
            selector = (hv.streams
                        .BoundsX(source=diagram)
                        .rename(boundsx=dim))
            self._diagrams[dim] = Plot(selector, diagram)

    def _conditions(self, **kwargs):
        """Извлекаем все уловия для подвыборки из статичных диаграмм"""
        conditions = np.repeat(True, len(self.data))  # Сначала задаем True

        for dim, value in kwargs.items():           # Название всех ограничений
            if dim in self._gdims:                  # совпадает с полями
                _, diagram = self._diagrams[dim]
                categories = diagram.data.loc[value][dim]
                if categories:
                    conditions &= self.data[dim].isin(categories)

            elif dim in self._ddims:
                _, diagram = self._diagrams[dim]
                if value:
                    left, right = value
                    left = pd.to_datetime(left, unit='ms')    # Переводим из
                    right = pd.to_datetime(right, unit='ms')  # млсек в даты
                    conditions &= self.data[dim].between(left, right)
        return conditions

    def _make_charts(self):
        """Создаем диаграммы вместе с меняющейся"""
        for dim in self._gdims + self._ddims:
            self.__dict__[dim] = (             # Добавляем диаграмму в атрибуты
                self._diagrams[dim].diagram *  # self. Небезопасно, если
                self._make_one_chart(dim))     # уже что-то есть с именем

    def _make_one_chart(self, dim):
        """Одна динамически обновляемая диаграмма"""

        selectors = [s for s, d in self._diagrams.values()]

        if dim in self._ddims:
            diagram_type = hv.Area
        elif dim in self._gdims:
            diagram_type = hv.Bars

        def bar_chart(**kwargs):
            data = self._get_count(dim, self._conditions(**kwargs))
            return diagram_type(data, kdims=[dim], vdims=['count'])

        return hv.DynamicMap(bar_chart, streams=selectors)

    def _make_isotinic_plot(self):
        """Создаем диаграмму с IR"""

        kdims = [hv.Dimension('predict', values=self._pdims),
                 hv.Dimension('target', values=self._tdims)]

        selectors = [s for s, d in self._diagrams.values()]

        def chart(target, predict, **kwargs):
            condisions = self._conditions(**kwargs)
            data = self.data.loc[condisions]
            return isotonic(data, predict, target, self._calibrations_data)

        iso_chart = hv.DynamicMap(chart, kdims=kdims, streams=selectors)
        self.__dict__['isotonic'] = iso_chart

    def _load_calibrations(self):
        if isinstance(self._calibrations_data, str):
            self.calibrations = pd.read_csv(self._calibrations_data)
        elif isinstance(self._calibrations_data, pd.DataFrame):
            self.calibrations = self._calibrations_data

    def _check_fields(self):
        """Проверяльщик формата"""
        assert isinstance(self.data, pd.DataFrame), 'data must be DataFrame'
        assert self._pdims is not None, '{} must be not None'.format('pdims')
        assert self._tdims is not None, '{} must be not None'.format('tdims')

        if self._ddims is None:
            self._ddims = []
        if self._gdims is None:
            self._gdims = []

        for dims in [self._gdims, self._tdims, self._ddims, self._pdims]:
            assert isinstance(dims, list), '{} must be list'.format(dims)
            for col in dims:
                assert col in self.data.columns, '{} must be a column of data'
