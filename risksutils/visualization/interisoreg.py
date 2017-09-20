from collections import namedtuple
from sklearn.isotonic import IsotonicRegression
from statsmodels.stats.proportion import proportion_confint
import holoviews as hv
import numpy as np
import pandas as pd


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
                selector, diagram = self._diagrams[dim]
                categories = diagram.data.loc[value][dim]
                if len(categories):
                    conditions &= self.data[dim].isin(categories) # Булев AND,
                                                                  # как +=

            elif dim in self._ddims:
                selector, diagram = self._diagrams[dim]
                if value:
                    left, right = value
                    left = pd.to_datetime(left, unit='ms')    # Переводим из
                    right = pd.to_datetime(right, unit='ms')  # млсек в даты
                    conditions &= self.data[dim].between(left, right)
        return conditions

    def _make_charts(self):
        """Создаем диаграммы вместе с меняющейся"""
        for dim in self._gdims + self._ddims:
            self.__dict__[dim] = (            # Добавляем диаграмму в атрибуты
                self._diagrams[dim].diagram   # self. Небезопасно, если с таким
              * self._make_one_chart(dim))    # же названием уже что-то есть

    def _make_one_chart(self, dim):
        """Одна динамически обновляемая диаграмма"""

        selectors = [s for s, d in self._diagrams.values()]

        if dim in self._ddims:
            diagram_type = hv.Area
        elif dim in self._gdims:
            diagram_type = hv.Bars

        def bar(**kwargs):
            data = self._get_count(dim, self._conditions(**kwargs))
            return diagram_type(data, kdims=[dim], vdims=['count'])

        return hv.DynamicMap(bar, streams=selectors)

    def _make_isotinic_plot(self):
        """Создаем диаграмму с IR"""

        kdims = [hv.Dimension('predict', values=self._pdims),
                 hv.Dimension('target', values=self._tdims)]

        selectors = [s for s, d in self._diagrams.values()]

        def chart(target, predict, **kwargs):
            condisions = self._conditions(**kwargs)
            data = self.data.loc[condisions]
            df = isotonic_plot_data(data, target, predict)
            confident_intervals = (hv.Area(df, kdims=['pred'],
                                           vdims=['ci_l', 'ci_h'])
                                     .opts(style=dict(alpha=0.5)))
            curve = hv.Curve(df, kdims=['pred'], vdims=['isotonic'])

            if self._calibrations_data is not None:
                if target in self.calibrations.columns:
                    calibr = hv.Curve(
                      data=self.calibrations[['pred', target]].values,
                      kdims=['pred'],
                      vdims=['target']
                    )
                    return confident_intervals * curve * calibr

            return confident_intervals * curve

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


def isotonic_plot_data(df, target, predict):
    """Подготавливаем данные для рисования Isotonic диаграммы"""
    reg = IsotonicRegression()
    return (df[[predict, target]]                 # выбираем только два поля
             .dropna()                            # оставляем только непустые
             .rename(columns={predict: 'pred',
                              target: 'target'})  # меняем их названия
             .assign(isotonic=lambda df:          # значение прогноза IR
                      reg.fit_transform(          # обучаем и считаем прогноз.
                          X=(df['pred']           # 🔫IR не работает с
                             + 1e-7 * np.random.rand(len(df))),
                          y=df['target']          # повторяющимися значениями
                      ))                          # поэтому костыльно делаем их
             .groupby('isotonic')                 # разными.
             .agg({'target': ['sum', 'count'],    # Для каждого значения ir
                   'pred': ['min', 'max']})       # агрегируем target
             .reset_index()
             .pipe(confident_intervals)           # доверительные интервалы
             .pipe(stack_min_max))                # Преобразуем в нужный формат


def confident_intervals(df):
    """Добавляем в таблицу доверительные интервалы"""
    df['ci_l'], df['ci_h'] = proportion_confint(
        count=df['target']['sum'],
        nobs=df['target']['count'],
        alpha=0.05,
        method='beta'
    )
    df['ci_l'] = df['ci_l'].fillna(0)
    df['ci_h'] = df['ci_h'].fillna(1)
    return df


def stack_min_max(df):
    """Перегруппировываем значения в таблице для последующего рисования"""
    stack = (df['pred']                      # pred - Мульти Индекс,
                                             # включающий min, max
               .stack()                      # Каждой строчке сопоставляем
                                             # две строчки со значениями
               .reset_index(1, drop=True)    # для min и для max,
               .rename('pred'))              # а потом меням название поля
    df = pd.concat([stack, df['isotonic'],
                    df['ci_l'], df['ci_h']], axis=1)
    df['ci_l'] = df['ci_l'].cummax()         # Делаем границы монотонными
    df['ci_h'] = df[::-1]['ci_h'].cummin()[::-1]
    return df
